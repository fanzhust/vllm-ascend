# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-side Engram storage, routing and lookup.

Tables are BF16, INT8/FP32 or native MXFP8/E8M0, sharded into contiguous hash
head buckets.  With ``EngramConfig.cpu_offload`` the shard stays in host memory:
``aclrtHostRegisterV2`` pins it and ``aclrtHostGetDevicePointer`` publishes the
address the NPU gather kernel reads, so an offloaded table needs neither an H2D
copy nor a host-side gather.
"""

import ctypes
import os
import time
from functools import cache
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.device.device_op import DeviceOperator

SCALE_GROUP = 32
MEMFABRIC_MAP_BLOCK = 256
# A 384M row table overflows the 32 bit offset arithmetic a single Triton tile
# can express, so the device address of every group of rows is published
# separately.
CHUNK_ROWS = 1 << 22

_MEMFABRIC_TRACE = os.getenv("VLLM_ASCEND_ENGRAM_MF_TRACE", "0").strip().lower() in ("1", "true", "yes", "on")
try:
    _MEMFABRIC_TRACE_CALLS = max(0, int(os.getenv("VLLM_ASCEND_ENGRAM_MF_TRACE_CALLS", "2")))
except ValueError:
    _MEMFABRIC_TRACE_CALLS = 2
_memfabric_trace_counts: dict[tuple[int, int], int] = {}
_MEMFABRIC_FENCE_STAGE = os.getenv("VLLM_ASCEND_ENGRAM_MF_FENCE_STAGE", "").strip().lower()
_MEMFABRIC_FENCE_STAGES = {"", "input", "map_ids", "entry_gather", "dequant", "all"}
if _MEMFABRIC_FENCE_STAGE not in _MEMFABRIC_FENCE_STAGES:
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_FENCE_STAGE must be one of "
        f"{sorted(_MEMFABRIC_FENCE_STAGES)}, got {_MEMFABRIC_FENCE_STAGE!r}"
    )
_MEMFABRIC_FENCE_SCOPE = os.getenv("VLLM_ASCEND_ENGRAM_MF_FENCE_SCOPE", "device").strip().lower()
if _MEMFABRIC_FENCE_SCOPE not in ("device", "probe"):
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_FENCE_SCOPE must be 'device' or 'probe', "
        f"got {_MEMFABRIC_FENCE_SCOPE!r}"
    )
_MEMFABRIC_SKIP_OPS = {
    op.strip().lower() for op in os.getenv("VLLM_ASCEND_ENGRAM_MF_SKIP_OPS", "").split(",") if op.strip()
}
_MEMFABRIC_SKIPPABLE_OPS = {"map_ids", "entry_gather", "dequant"}
if unknown_skip_ops := _MEMFABRIC_SKIP_OPS - _MEMFABRIC_SKIPPABLE_OPS:
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_SKIP_OPS contains unsupported operators: "
        f"{sorted(unknown_skip_ops)}; supported={sorted(_MEMFABRIC_SKIPPABLE_OPS)}"
    )


def engram_cpu_offload(vllm_config) -> bool:
    """Whether the Engram table is offloaded to host memory (UVA lookup).

    ``--engram-config`` turns on host offload. Without it, the tables stay on
    the device, exactly like upstream.
    """

    engram_config = getattr(vllm_config, "engram_config", None)
    return bool(engram_config is not None and engram_config.cpu_offload)


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    grouped = rows.float().unflatten(-1, (-1, SCALE_GROUP))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    # NPU exp2 can return one ULP below an exact power of two, changing
    # ties-to-even codes. ldexp constructs the binary scale exactly.
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows(codes, scale):
    # Keep one FP32 work buffer: in-place scaling avoids the extra FP32 result
    # allocation created by the broadcast multiply expression.
    decoded = codes.float().unflatten(-1, (-1, SCALE_GROUP))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


def gather_dequantize_engram_int8(
    weight: torch.Tensor,
    scales: torch.Tensor | None,
    ids: torch.Tensor,
    width: int,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather device rows, dequantizing INT8 or MXFP8 when scales are supplied.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """

    # Importing the ops package initializes the active Triton backend, so keep
    # it out of CPU-only routing and test workers.
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    vocab_end = weight.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=weight.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    from vllm_ascend.ops.triton.engram_lookup import _engram_int8_gather_dequant_kernel

    _engram_int8_gather_dequant_kernel[(rows,)](
        weight,
        scales,
        ids,
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        QUANTIZED=scales is not None,
        MXFP8=weight.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )
    return output


@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""

    def __init__(self, shape, dtype, device):
        self.lib = _host_library()
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        try:
            rc = self.lib.aclrtHostRegisterV2(self.pointer, size, DeviceOperator.host_register_flags())
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception:
            # A half-built buffer must not leave the host range registered: the
            # caller only sees the exception, so nothing else can release it.
            self.lib.aclrtHostUnregister(self.pointer)
            self.lib.aclrtFreeHost(self.pointer)
            self.tensor = None
            self.buffer = None
            self.pointer = ctypes.c_void_p()
            raise

    def close(self):
        """Unregister and release the host range; safe to call more than once.

        The NPU gathers out of this range through the device address the
        registration published, so the mapping must not go away while work that
        reads it is still in flight.
        """
        if self.pointer is None or not self.pointer.value:
            return
        if self.ptrs is not None and self.ptrs.device.type == "npu":
            torch.npu.synchronize()
        rc = self.lib.aclrtHostUnregister(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
        self.tensor = None
        self.buffer = None
        self.ptrs = None
        rc = self.lib.aclrtFreeHost(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtFreeHost failed: rc={rc}")
        self.pointer = ctypes.c_void_p()


def gather_dequantize_host_uva(
    codes: HostUvaBuffer,
    scales: HostUvaBuffer | None,
    ids: torch.Tensor,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather registered host rows, preserving BF16 when scales are absent.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    width = codes.tensor.shape[-1]
    vocab_end = codes.tensor.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    # The kernel widens loaded IDs to int64; no extra cast/copy is needed here.
    from vllm_ascend.ops.triton.engram_lookup import _engram_host_uva_gather_dequant_kernel

    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs if scales is not None else None,
        ids,
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        CHUNK=CHUNK_ROWS,
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        QUANTIZED=scales is not None,
        MXFP8=codes.tensor.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )
    return output


def gather_dequantize_memfabric(
    table,
    ids: torch.Tensor,
    *,
    head_start: int,
    local_heads: int,
    pad_heads: int,
    output: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> torch.Tensor:
    """Gather packed 288-byte INT8/FP32 rows and dequantize on the NPU."""
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    rows = ids.shape[0] * local_heads
    if rows == 0:
        return output
    if rows > table.global_ids.numel():
        raise ValueError(
            f"MemFabric Engram lookup needs {rows} rows, but its stable scratch holds {table.global_ids.numel()}"
        )
    # Diagnostic-only producer fence.  Unlike a device-wide synchronize this
    # waits only for the upstream chain that materializes the hash ids, which
    # distinguishes an unavailable input from a stuck map_ids launch.
    _memfabric_debug_fence(table, "input", ids)
    init_device_properties_triton()
    from vllm_ascend.ops.triton.engram_lookup import (
        _engram_memfabric_dequant_kernel,
        _engram_memfabric_map_ids_kernel,
    )

    map_grid = ((rows + MEMFABRIC_MAP_BLOCK - 1) // MEMFABRIC_MAP_BLOCK,)
    trace_call = None
    if _MEMFABRIC_TRACE:
        trace_call = _begin_memfabric_trace(
            table,
            ids,
            rows=rows,
            head_start=head_start,
            local_heads=local_heads,
            pad_heads=pad_heads,
            vocab_start=vocab_start,
            vocab_end=vocab_end,
        )
        _memfabric_trace_event(
            table,
            trace_call,
            f"map_ids launch begin grid={map_grid[0]} block={MEMFABRIC_MAP_BLOCK}",
        )
    if "map_ids" in _MEMFABRIC_SKIP_OPS:
        # Keep entry_gather valid while removing the Triton mapper. Repeating
        # this rank's first row avoids remote traffic and remains inside this
        # layer's registered segment.
        safe_global_id = table.rank_id * table.rows_per_slot + table.layer_base
        table.global_ids[:rows].fill_(safe_global_id)
        table.count.fill_(rows)
        _memfabric_skip_event(table, "map_ids", f"global_id={safe_global_id} count={rows}")
    else:
        _engram_memfabric_map_ids_kernel[map_grid](
            ids,
            table.global_ids,
            table.count,
            rows,
            vocab_start,
            vocab_end,
            ids.stride(0),
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            SLOT_RANK=table.rank_id,
            LAYER_BASE=table.layer_base,
            ROWS_PER_SLOT=table.rows_per_slot,
            BLOCK=MEMFABRIC_MAP_BLOCK,
            num_warps=1,
        )
        if trace_call is not None:
            _memfabric_trace_event(table, trace_call, "map_ids submitted")
    _memfabric_debug_fence(table, "map_ids", table.count, expected=rows)

    gather_started = time.perf_counter() if trace_call is not None else 0.0
    if "entry_gather" in _MEMFABRIC_SKIP_OPS:
        # A packed all-zero row has zero codes and zero FP32 scales, so the
        # real dequant kernel can safely consume it and produce zero output.
        table.staging[:rows].zero_()
        _memfabric_skip_event(table, "entry_gather", f"zeroed_rows={rows}")
    else:
        if trace_call is not None:
            _memfabric_trace_event(table, trace_call, "entry_gather call begin")
        rc = _submit_memfabric_entry_gather(table)
        if trace_call is not None:
            _memfabric_trace_event(
                table,
                trace_call,
                f"entry_gather returned rc={rc} host_elapsed_ms={(time.perf_counter() - gather_started) * 1000:.3f}",
            )
        if rc:
            raise RuntimeError(f"MemFabric Engram entry_gather failed: rc={rc}")
    _memfabric_debug_fence(table, "entry_gather", table.staging)
    if "dequant" in _MEMFABRIC_SKIP_OPS:
        output.zero_()
        _memfabric_skip_event(table, "dequant", f"zeroed_output_shape={tuple(output.shape)}")
    else:
        if trace_call is not None:
            _memfabric_trace_event(table, trace_call, "dequant launch begin")
        _engram_memfabric_dequant_kernel[(rows,)](
            table.staging,
            table.staging_fp32,
            ids,
            output,
            rows,
            vocab_start,
            vocab_end,
            ids.stride(0),
            ENTRY_BYTES=table.staging.shape[1],
            CODE_BYTES=table.code_bytes,
            GROUP=SCALE_GROUP,
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            PAD_HEADS=pad_heads,
            num_warps=4,
        )
        if trace_call is not None:
            _memfabric_trace_event(table, trace_call, "dequant submitted")
            _memfabric_trace_event(table, trace_call, "lookup submitted")
    _memfabric_debug_fence(table, "dequant", output)
    return output


def _begin_memfabric_trace(
    table,
    ids: torch.Tensor,
    *,
    rows: int,
    head_start: int,
    local_heads: int,
    pad_heads: int,
    vocab_start: int,
    vocab_end: int,
) -> int | None:
    """Start one opt-in eager lookup trace."""
    if not _MEMFABRIC_TRACE or _MEMFABRIC_TRACE_CALLS == 0:
        return None
    layer = table.segment.layer_hash_index
    key = (table.pool.topology.device_id, layer)
    previous_calls = _memfabric_trace_counts.get(key, 0)
    if previous_calls >= _MEMFABRIC_TRACE_CALLS:
        return None
    call = previous_calls + 1
    _memfabric_trace_counts[key] = call
    logger.warning(
        "[Engram MF trace] BEGIN device=%d mf_rank=%d layer=%d call=%d "
        "tokens=%d rows=%d ids_shape=%s ids_dtype=%s ids_stride=%s "
        "heads=%d:%d pad_heads=%d vocab=[%d,%d) capacity=%d "
        "layer_base=%d rows_per_slot=%d staging=0x%x staging_fp32=0x%x "
        "staging_shape=%s",
        table.pool.topology.device_id,
        table.rank_id,
        layer,
        call,
        ids.shape[0],
        rows,
        tuple(ids.shape),
        ids.dtype,
        tuple(ids.stride()),
        head_start,
        head_start + local_heads,
        pad_heads,
        vocab_start,
        vocab_end,
        table.capacity,
        table.layer_base,
        table.rows_per_slot,
        table.staging.data_ptr(),
        table.staging_fp32.data_ptr(),
        tuple(table.staging.shape),
    )
    return call


def _memfabric_trace_event(table, call: int | None, event: str) -> None:
    if call is None:
        return
    logger.warning(
        "[Engram MF trace] device=%d mf_rank=%d layer=%d call=%d %s",
        table.pool.topology.device_id,
        table.rank_id,
        table.segment.layer_hash_index,
        call,
        event,
    )


def _memfabric_skip_event(table, op: str, detail: str) -> None:
    logger.warning(
        "[Engram MF skip] device=%d mf_rank=%d layer=%d op=%s %s",
        table.pool.topology.device_id,
        table.rank_id,
        table.segment.layer_hash_index,
        op,
        detail,
    )


def _memfabric_debug_fence(
    table,
    stage: str,
    probe: torch.Tensor,
    *,
    expected: int | None = None,
) -> None:
    """Optionally fence one MF stage to locate cross-stream stalls.

    Device scope verifies every stream. Probe scope copies one result scalar
    to the host, fencing only the producer chain of that result and avoiding
    unrelated HCCL streams. It is disabled unless explicitly requested.
    """
    if _MEMFABRIC_FENCE_STAGE not in (stage, "all"):
        return
    topology = table.pool.topology
    started = time.perf_counter()
    prefix = (
        f"[Engram MF fence] device={topology.device_id} mf_rank={table.rank_id} "
        f"layer={table.segment.layer_hash_index} stage={stage}"
    )
    if _MEMFABRIC_FENCE_SCOPE == "probe":
        logger.warning("%s tensor probe BEGIN", prefix)
        value = probe.reshape(-1)[0].item()
        if expected is not None and value != expected:
            raise RuntimeError(f"{prefix} tensor probe mismatch: value={value}, expected={expected}")
        logger.warning(
            "%s tensor probe DONE value=%s elapsed_ms=%.3f",
            prefix,
            value,
            (time.perf_counter() - started) * 1000,
        )
    else:
        logger.warning("%s device synchronize BEGIN", prefix)
        torch.npu.synchronize(topology.device_id)
        logger.warning(
            "%s device synchronize DONE elapsed_ms=%.3f",
            prefix,
            (time.perf_counter() - started) * 1000,
        )


def _submit_memfabric_entry_gather(table) -> int:
    """Submit through MemFabric's public tensor-oriented Python API."""
    return table.pool.offload.entry_gather(
        table.staging,
        table.global_ids,
        table.count,
        table.staging.device,
    )


class SharedUvaBuffer:
    """One host range mapped and registered by every rank of a local group.

    The leader creates the SharedMemory segment; every rank registers its own
    mapping with CANN. The leader unlinks it after all ranks have attached.
    """

    def __init__(self, shape, dtype, device, group):
        self.lib = _host_library()
        self.group = group
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.shm = None
        self.tensor = None
        self.ptrs = None
        self.pointer = ctypes.c_void_p()

        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                payload = [self.shm.name]
            except Exception as exc:  # noqa: BLE001 - reported to the group
                if self.shm is not None:
                    self.shm.unlink()
                    self.shm.close()
                payload = [f"ERROR: {type(exc).__name__}: {exc}"]
        dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        name = payload[0]
        if name is None or name.startswith("ERROR:"):
            raise RuntimeError(f"Engram shared backing creation failed: {name}")

        error = None
        try:
            if self.shm is None:
                # Python 3.12 tracks attachments as owners. Match vLLM's
                # SharedMemory attach path so only the creator unlinks it.
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    self.shm = shared_memory.SharedMemory(name=name)
            assert self.shm.size >= size
            address_of_mapping = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)))
            rc = self.lib.aclrtHostRegisterV2(address_of_mapping, size, DeviceOperator.host_register_flags())
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            # Registration succeeded: from here on the mapping owes exactly one
            # aclrtHostUnregister, and close() must keep the owner until it
            # returns success.  ``pointer`` is therefore set only
            # after the registration it has to undo exists.
            self.pointer = address_of_mapping
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=rows * row_elements).reshape(shape)
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        errors: list[str | None] = [None] * group.world_size
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence every mapping (or every failure) before the leader unlinks.
        dist.barrier(group=cpu_group)
        if failures:
            try:
                self.close()
            except RuntimeError as exc:  # a rank that did register still owes it
                failures = f"{failures}; release failed: {exc}"
            if group.rank_in_group == 0:
                self._unlink()
            raise RuntimeError(f"Engram shared-memory initialization failed: {failures}")
        if group.rank_in_group == 0:
            self._unlink()

    def _unlink(self) -> None:
        try:
            shm = self.shm
            if shm is not None:
                shm.unlink()
        except OSError as exc:
            logger.warning("Engram shared backing unlink failed: %s", exc)

    def close(self) -> None:
        """Unregister and close the mapping; shared memory is never aclrtFreeHost.

        Safe to call more than once.  A failed unregister raises and keeps the
        pointer, the CPU views and the device address table, so the caller can
        retry: the backing stays owned until CANN has really let go of it, the
        same contract as the private ``HostUvaBuffer``.  The
        mapping is only closed after the unregister succeeded -- or when
        there is nothing registered to begin with, e.g. after a failed
        ``aclrtHostRegisterV2``.
        """
        if self.pointer is not None and self.pointer.value:
            if self.ptrs is not None and self.ptrs.device.type == "npu":
                torch.npu.synchronize()
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                raise RuntimeError(f"aclrtHostUnregister failed for shared Engram: rc={rc}")
        self.pointer = ctypes.c_void_p()
        self.tensor = None
        self.ptrs = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None
