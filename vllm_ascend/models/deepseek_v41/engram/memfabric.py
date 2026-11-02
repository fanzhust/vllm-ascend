# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP-scoped MemFabric storage for DeepSeek-V4.1 Engram tables.

Every DP replica owns one SHARED pool containing all TP ranks. A pool slot is
the complete local head shard of its TP rank; rows are never partitioned again
over DP. All Engram layers are folded into one uniform 288-byte row grid (256
INT8 code bytes followed by 8 FP32 scales) because acc_offload exposes one
registered entry-table layout per process. Embeddings receive lightweight
table views; they never initialize or release the pool themselves.
"""

from __future__ import annotations

import ctypes
import os
import socket
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import logger

CODE_BYTES = 256
SCALE_COUNT = 8
SCALE_BYTES = SCALE_COUNT * 4
ENTRY_BYTES = CODE_BYTES + SCALE_BYTES
ONE_GIB = 1 << 30
MAX_ENTRY_ID = 1 << 32

_MEMFABRIC_TRACE = os.getenv("VLLM_ASCEND_ENGRAM_MF_TRACE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MEMFABRIC_INIT_PROBE = os.getenv("VLLM_ASCEND_ENGRAM_MF_INIT_PROBE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MEMFABRIC_DIAG_LOCAL = os.getenv("VLLM_ASCEND_ENGRAM_MF_DIAG_LOCAL", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MEMFABRIC_DIAG_SHARED_SINGLE = os.getenv("VLLM_ASCEND_ENGRAM_MF_DIAG_SHARED_SINGLE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MEMFABRIC_REPLICATED = os.getenv("VLLM_ASCEND_ENGRAM_MF_REPLICATED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MEMFABRIC_GVA_MODE = os.getenv("VLLM_ASCEND_ENGRAM_MF_GVA_MODE", "full").strip().lower()
if _MEMFABRIC_GVA_MODE not in ("full", "minimal", "none"):
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_GVA_MODE must be one of: full, minimal, none; "
        f"got {_MEMFABRIC_GVA_MODE!r}"
    )
if sum((_MEMFABRIC_DIAG_LOCAL, _MEMFABRIC_DIAG_SHARED_SINGLE, _MEMFABRIC_REPLICATED)) > 1:
    raise ValueError(
        "MemFabric LOCAL diagnostic, SHARED-single-rank diagnostic and replicated modes are mutually exclusive"
    )
if _MEMFABRIC_GVA_MODE != "full":
    if _MEMFABRIC_DIAG_LOCAL or _MEMFABRIC_DIAG_SHARED_SINGLE or _MEMFABRIC_REPLICATED:
        raise ValueError("MemFabric minimal/none GVA modes cannot be combined with scene or replicated modes")
    bypass = os.getenv("VLLM_ASCEND_ENGRAM_MF_BYPASS", "").strip().lower()
    skip_load = os.getenv("VLLM_ASCEND_ENGRAM_MF_SKIP_LOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if bypass != "full" or not skip_load:
        raise ValueError(
            "MemFabric minimal/none GVA modes require "
            "VLLM_ASCEND_ENGRAM_MF_BYPASS=full and VLLM_ASCEND_ENGRAM_MF_SKIP_LOAD=1"
        )
if (_MEMFABRIC_DIAG_LOCAL or _MEMFABRIC_DIAG_SHARED_SINGLE) and os.getenv(
    "VLLM_ASCEND_ENGRAM_MF_BYPASS", ""
).strip().lower() != "full":
    raise ValueError(
        "MemFabric scene diagnostics require VLLM_ASCEND_ENGRAM_MF_BYPASS=full; "
        "they preserve model loading only and must not execute entry_gather"
    )


def _load_acl_context_api() -> ctypes.CDLL:
    """Load the two ACL APIs needed to preserve the caller's context.

    HYBM initializes its device with ``aclrtSetDevice``.  Engram constructs
    MemFabric after torch_npu has already established its context and streams,
    so MemFabric must not leave a different context current on this thread.
    """
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtGetCurrentContext.restype = ctypes.c_int
    lib.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
    lib.aclrtSetCurrentContext.restype = ctypes.c_int
    return lib


def _get_acl_current_context(lib: ctypes.CDLL) -> int | None:
    context = ctypes.c_void_p()
    rc = lib.aclrtGetCurrentContext(ctypes.byref(context))
    if rc:
        raise RuntimeError(f"aclrtGetCurrentContext failed: rc={rc}")
    return context.value


def _restore_acl_context(lib: ctypes.CDLL, context: int | None) -> tuple[int | None, bool]:
    """Restore a non-null caller context and return (observed, changed)."""
    observed = _get_acl_current_context(lib)
    changed = observed != context
    if changed and context is not None:
        rc = lib.aclrtSetCurrentContext(ctypes.c_void_p(context))
        if rc:
            raise RuntimeError(
                "aclrtSetCurrentContext failed while restoring the torch_npu "
                f"context: rc={rc}, saved=0x{context:x}, "
                f"observed={None if observed is None else hex(observed)}"
            )
    return observed, changed


def _probe_tp_after_memfabric_init_stage(
    topology: MemFabricEngramTopology,
    stage: str,
    acl: ctypes.CDLL,
    saved_context: int | None,
) -> None:
    """Force one tiny TP collective to locate MF initialization side effects."""
    if not _MEMFABRIC_INIT_PROBE:
        return

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    current_context = _get_acl_current_context(acl)
    logger.warning(
        "[Engram MF init probe] stage=%s BEGIN device=%d mf_rank=%d tp_rank=%d "
        "tp_size=%d saved_context=%s current_context=%s",
        stage,
        topology.device_id,
        topology.rank_id,
        topology.tp_rank,
        tp_group.world_size,
        "NULL" if saved_context is None else f"0x{saved_context:x}",
        "NULL" if current_context is None else f"0x{current_context:x}",
    )
    started = time.perf_counter()
    value = torch.ones(1, dtype=torch.float32, device=torch.device("npu", topology.device_id))
    dist.all_reduce(value, group=tp_group.device_group)
    observed = float(value.item())
    expected = float(tp_group.world_size)
    if observed != expected:
        raise RuntimeError(
            f"MemFabric init TP probe mismatch at {stage}: observed={observed}, expected={expected}"
        )
    logger.warning(
        "[Engram MF init probe] stage=%s DONE device=%d mf_rank=%d tp_rank=%d "
        "value=%.1f elapsed_ms=%.3f",
        stage,
        topology.device_id,
        topology.rank_id,
        topology.tp_rank,
        observed,
        (time.perf_counter() - started) * 1000,
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


@dataclass(frozen=True)
class MemFabricEngramSegment:
    layer_hash_index: int
    rows: int
    capacity: int
    layer_base: int


@dataclass(frozen=True)
class MemFabricEngramLayout:
    world_size: int
    rows_per_slot: int
    local_rows_span: int
    segments: tuple[MemFabricEngramSegment, ...]

    @classmethod
    def build(
        cls,
        layer_rows: list[tuple[int, int]],
        world_size: int,
        *,
        rank_id: int = 0,
        rank_layer_rows: list[list[tuple[int, int]]] | None = None,
    ) -> MemFabricEngramLayout:
        """Build one uniform TP-slot grid while retaining local row counts.

        Every rank registers the same ``rows_per_slot``. A layer therefore
        receives the maximum local-row count of that layer over the TP group,
        while ``segment.rows`` remains this rank's actual local shard size.
        """
        if world_size <= 0:
            raise ValueError("MemFabric Engram world_size must be positive")
        if not 0 <= rank_id < world_size:
            raise ValueError(f"MemFabric Engram rank_id {rank_id} is outside world_size {world_size}")
        if rank_layer_rows is None:
            if world_size != 1:
                raise ValueError("TP-scoped MemFabric layout needs every rank's layer rows when world_size > 1")
            rank_layer_rows = [layer_rows]
        if len(rank_layer_rows) != world_size:
            raise ValueError(
                "MemFabric Engram rank-layer layout size mismatch: "
                f"got {len(rank_layer_rows)}, expected {world_size}"
            )
        if rank_layer_rows[rank_id] != layer_rows:
            raise ValueError(
                f"MemFabric Engram local layer rows do not match TP pool rank {rank_id}: "
                f"local={layer_rows}, gathered={rank_layer_rows[rank_id]}"
            )
        layer_ids = [layer_hash_index for layer_hash_index, _ in layer_rows]
        for slot_rank, slot_rows in enumerate(rank_layer_rows):
            if [layer_hash_index for layer_hash_index, _ in slot_rows] != layer_ids:
                raise ValueError(
                    "MemFabric Engram layers must match across TP pool ranks: "
                    f"rank0={layer_ids}, rank{slot_rank}={slot_rows}"
                )

        layer_base = 0
        segments = []
        for layer_position, (layer_hash_index, rows) in enumerate(layer_rows):
            if rows <= 0:
                raise ValueError(f"Engram layer {layer_hash_index} must have at least one row")
            rank_counts = [slot_rows[layer_position][1] for slot_rows in rank_layer_rows]
            if any(rank_rows <= 0 for rank_rows in rank_counts):
                raise ValueError(
                    f"Engram layer {layer_hash_index} must have at least one row on every TP rank; "
                    f"got {rank_counts}"
                )
            capacity = max(rank_counts)
            segments.append(MemFabricEngramSegment(layer_hash_index, rows, capacity, layer_base))
            layer_base += capacity
        if not segments:
            raise ValueError("MemFabric Engram needs at least one table")
        if world_size * layer_base >= MAX_ENTRY_ID:
            raise ValueError(
                f"MemFabric Engram entry grid exceeds uint32 IDs: world_size={world_size}, rows_per_slot={layer_base}"
            )
        local_rows_span = max(segment.layer_base + segment.rows for segment in segments)
        return cls(
            world_size=world_size,
            rows_per_slot=layer_base,
            local_rows_span=local_rows_span,
            segments=tuple(segments),
        )

    def segment(self, layer_hash_index: int) -> MemFabricEngramSegment:
        for segment in self.segments:
            if segment.layer_hash_index == layer_hash_index:
                return segment
        raise KeyError(f"Unknown MemFabric Engram layer index {layer_hash_index}")


@dataclass(frozen=True)
class MemFabricEngramTopology:
    rank_id: int
    world_size: int
    local_world_size: int
    device_id: int
    pool_index: int
    store_url: str
    replicated: bool = False
    tp_rank: int = 0
    pool_scope: str = "tp_group"

    @staticmethod
    def _group_topology(group, description: str) -> tuple[int, int, int]:
        rank_id = group.rank_in_group
        world_size = group.world_size
        hostnames: list[str | None] = [None] * world_size
        dist.all_gather_object(hostnames, socket.gethostname(), group=group.cpu_group)
        if any(host is None for host in hostnames):
            raise RuntimeError(f"MemFabric Engram failed to collect {description} host topology")
        hosts = [str(host) for host in hostnames]
        counts = {host: hosts.count(host) for host in set(hosts)}
        if len(set(counts.values())) != 1:
            raise ValueError(f"MemFabric Engram requires equal local pool sizes; got {counts}")
        seen: set[str] = set()
        previous = None
        for host in hosts:
            if host != previous:
                if host in seen:
                    raise ValueError(
                        f"MemFabric Engram {description} ranks must be machine-major; got host order {hosts}"
                    )
                seen.add(host)
                previous = host
        return rank_id, world_size, counts[hosts[rank_id]]

    @classmethod
    def from_runtime(cls, config) -> MemFabricEngramTopology:
        pool_scope = getattr(config, "pool_scope", "tp_group")
        if pool_scope != "tp_group":
            raise ValueError(
                "MemFabric Engram only supports engram_memfabric.pool_scope=tp_group; "
                f"got {pool_scope!r}"
            )
        tp_rank = get_tensor_model_parallel_rank()
        one_rank_pool = _MEMFABRIC_DIAG_SHARED_SINGLE or _MEMFABRIC_REPLICATED
        if one_rank_pool:
            rank_id = 0
            world_size = 1
            local_world_size = 1
            pool_index = tp_rank
        else:
            tp_group = get_tp_group()
            rank_id, world_size, local_world_size = cls._group_topology(tp_group, "TP-group")
            if rank_id != tp_rank:
                raise ValueError(
                    "MemFabric TP pool rank must equal the tensor-parallel rank: "
                    f"pool_rank={rank_id}, tp_rank={tp_rank}"
                )
            # One TP pool belongs to each DP replica. The DP rank therefore
            # selects its store, while the TP rank identifies the participant
            # inside that pool.
            pool_index = get_dp_group().rank_in_group
            pool_indices: list[int | None] = [None] * world_size
            dist.all_gather_object(pool_indices, pool_index, group=tp_group.cpu_group)
            if any(index != pool_index for index in pool_indices):
                raise ValueError(
                    "All TP ranks in one MemFabric pool must have the same DP rank; "
                    f"got {pool_indices}"
                )

        device_id = torch.npu.current_device()
        # A one-rank pool needs one independent rank-0 store per worker. Use
        # the unique local device id as its store slot; normal DP sharing
        # groups continue to use one store per TP rank.
        store_slot = device_id if one_rank_pool else pool_index
        port = config.store_port_base + store_slot
        if port > 65535:
            raise ValueError(
                f"MemFabric Engram store port exceeds 65535: base={config.store_port_base}, store_slot={store_slot}"
            )
        store_url = config.store_url_template.format(port=port, pool_index=store_slot, tp_rank=tp_rank)
        if len(store_url.encode()) >= 64:
            raise ValueError(f"MemFabric Engram store URL must be shorter than 64 bytes: {store_url!r}")
        return cls(
            rank_id=rank_id,
            world_size=world_size,
            local_world_size=local_world_size,
            device_id=device_id,
            pool_index=pool_index,
            store_url=store_url,
            replicated=_MEMFABRIC_REPLICATED,
            tp_rank=tp_rank,
            pool_scope=pool_scope,
        )


def _load_offload_module():
    try:
        from memfabric_hybrid import offload
    except ImportError as exc:
        raise RuntimeError(
            "MemFabric Engram offload requires the memfabric_hybrid Python package and acc_offload shared library"
        ) from exc
    return offload


class MemFabricEngramTableView:
    """One layer segment plus stable NPU gather scratch."""

    def __init__(
        self,
        pool: MemFabricEngramPool,
        segment: MemFabricEngramSegment,
        max_lookup_rows: int,
        device: torch.device,
        lookup_output_shape: tuple[int, int, int] | None = None,
    ) -> None:
        self.pool = pool
        self.segment = segment
        self.rank_id = pool.topology.rank_id
        self.rows_per_slot = pool.layout.rows_per_slot
        self.code_bytes = CODE_BYTES
        # A TP pool slot stores one complete local head shard. ``capacity`` is
        # only the uniform per-layer spacing shared by all slots; it is not a
        # second ownership partition over this rank's rows.
        self.owned_start = 0
        self.owned_end = segment.rows
        self.local_entries = pool.entries[segment.layer_base : segment.layer_base + segment.rows]
        self.global_ids = torch.empty(max_lookup_rows, dtype=torch.int64, device=device)
        self.count = torch.empty(1, dtype=torch.int32, device=device)
        # entry_gather requires the packed destination address to be aligned to
        # the row pitch. 288 is not a power of two and is not an alignment the
        # NPU caching allocator promises, so retain an over-allocation and take
        # an explicitly aligned, stable view from it.
        staging_bytes = max_lookup_rows * ENTRY_BYTES
        self._staging_storage = torch.empty(staging_bytes + ENTRY_BYTES - 1, dtype=torch.int8, device=device)
        staging_offset = (-self._staging_storage.data_ptr()) % ENTRY_BYTES
        self.staging = self._staging_storage[staging_offset : staging_offset + staging_bytes].view(
            max_lookup_rows, ENTRY_BYTES
        )
        if self.staging.data_ptr() % ENTRY_BYTES:
            raise RuntimeError(
                f"MemFabric Engram staging buffer is not {ENTRY_BYTES}-byte aligned: "
                f"address=0x{self.staging.data_ptr():x}"
            )
        # Ascend Triton rejects casting an int8 pointer directly to an FP32
        # pointer. Keep a typed, zero-copy alias of the same packed rows so the
        # dequant kernel can load the trailing scales without a pointer cast.
        # ENTRY_BYTES and the explicitly aligned base are both FP32-aligned.
        self.staging_fp32 = self.staging.view(torch.float32)
        self.lookup_output = None
        if lookup_output_shape is not None:
            if lookup_output_shape[2] != CODE_BYTES:
                raise ValueError(
                    f"MemFabric Engram lookup output width must be {CODE_BYTES}, got {lookup_output_shape}"
                )
            if _MEMFABRIC_TRACE:
                stream = pool.stream
                logger.warning(
                    "[Engram MF trace] SCRATCH allocation begin device=%d mf_rank=%d layer=%d "
                    "shape=%s dtype=%s stream=0x%x allocated_mib=%.2f reserved_mib=%.2f",
                    pool.topology.device_id,
                    self.rank_id,
                    segment.layer_hash_index,
                    lookup_output_shape,
                    torch.bfloat16,
                    pool.stream_handle,
                    torch.npu.memory_allocated() / (1 << 20),
                    torch.npu.memory_reserved() / (1 << 20),
                )
            self.lookup_output = torch.empty(lookup_output_shape, dtype=torch.bfloat16, device=device)
            if _MEMFABRIC_TRACE:
                stream = pool.stream
                logger.warning(
                    "[Engram MF trace] SCRATCH allocation returned device=%d mf_rank=%d layer=%d "
                    "ptr=0x%x stream=0x%x allocated_mib=%.2f reserved_mib=%.2f",
                    pool.topology.device_id,
                    self.rank_id,
                    segment.layer_hash_index,
                    self.lookup_output.data_ptr(),
                    pool.stream_handle,
                    torch.npu.memory_allocated() / (1 << 20),
                    torch.npu.memory_reserved() / (1 << 20),
                )
                logger.warning(
                    "[Engram MF trace] SCRATCH allocation synchronize begin "
                    "device=%d mf_rank=%d layer=%d",
                    pool.topology.device_id,
                    self.rank_id,
                    segment.layer_hash_index,
                )
                started = time.perf_counter()
                stream.synchronize()
                logger.warning(
                    "[Engram MF trace] SCRATCH allocation synchronize done "
                    "device=%d mf_rank=%d layer=%d elapsed_ms=%.3f",
                    pool.topology.device_id,
                    self.rank_id,
                    segment.layer_hash_index,
                    (time.perf_counter() - started) * 1000,
                )
        self.closed = False

    @property
    def capacity(self) -> int:
        return self.segment.capacity

    @property
    def layer_base(self) -> int:
        return self.segment.layer_base

    def copy_entries(self, local_row_start: int, codes: torch.Tensor, scales: torch.Tensor) -> None:
        """Copy local TP-shard rows as packed [256B codes, 32B scales]."""
        if codes.ndim != 2 or codes.shape[1] != CODE_BYTES:
            raise ValueError(f"MemFabric Engram codes must have shape [rows, {CODE_BYTES}], got {tuple(codes.shape)}")
        if codes.dtype != torch.int8:
            raise ValueError(f"MemFabric Engram codes must be INT8, got {codes.dtype}")
        if scales.ndim != 2 or scales.shape != (codes.shape[0], SCALE_COUNT):
            raise ValueError(
                f"MemFabric Engram scales must have shape [rows, {SCALE_COUNT}], got {tuple(scales.shape)}"
            )
        if scales.dtype != torch.float32:
            raise ValueError(f"MemFabric Engram scales must be FP32, got {scales.dtype}")
        local_row_end = local_row_start + codes.shape[0]
        if local_row_start < 0 or local_row_end > self.segment.rows:
            raise ValueError(
                "MemFabric Engram local row write is outside the TP shard: "
                f"range=[{local_row_start}, {local_row_end}), rows={self.segment.rows}"
            )
        rows = codes.shape[0]
        target = self.local_entries[local_row_start:local_row_end]
        source_codes = codes.contiguous().view(torch.uint8)
        source_scales = scales.contiguous().view(torch.uint8)
        target[:, :CODE_BYTES].copy_(source_codes.reshape(rows, CODE_BYTES))
        target[:, CODE_BYTES:].copy_(source_scales.reshape(rows, SCALE_BYTES))

    def close(self) -> None:
        if self.closed:
            return
        self.local_entries = None
        self.global_ids = None
        self.count = None
        self.staging_fp32 = None
        self.staging = None
        self._staging_storage = None
        self.lookup_output = None
        self.pool = None
        self.closed = True


class MemFabricEngramPool:
    """One process-local participant in a SHARED pool or LOCAL diagnostic."""

    def __init__(self, topology: MemFabricEngramTopology, layout: MemFabricEngramLayout) -> None:
        self.topology = topology
        self.layout = layout
        self.offload: Any | None = None
        self.pointer = 0
        self.entries = None
        self._buffer = None
        self.diagnostic_local = _MEMFABRIC_DIAG_LOCAL
        self.diagnostic_shared_single = _MEMFABRIC_DIAG_SHARED_SINGLE
        self.replicated = topology.replicated
        self.gva_mode = _MEMFABRIC_GVA_MODE
        # Capture the torch_npu stream before HYBM initializes its own ACL
        # runtime state. Forward tracing must reuse this object instead of
        # asking ACL for the current stream again after MemFabric setup.
        self.stream = torch.npu.current_stream(topology.device_id)
        self.stream_handle = int(self.stream.npu_stream)
        self.initialized = False
        self.closed = False

        reserve_bytes = layout.rows_per_slot * ENTRY_BYTES
        local_grid_bytes = layout.local_rows_span * ENTRY_BYTES
        self.reserve_size = _align_up(reserve_bytes, ONE_GIB)
        # register_entry_table requires the complete uniform row grid to fit
        # every rank's physically backed span, even when a rank has fewer
        # useful local rows. Keep alloc_size equal to the common slot reserve.
        self.allocation_size = self.reserve_size
        self.mapped_reserve_size = self.reserve_size
        if self.gva_mode == "none":
            # Keep every model-side MemFabric object and NPU scratch allocation,
            # but do not touch acc_offload/HYBM. Empty CPU storage is safe only
            # with the import-time skip-load + full-bypass guard above.
            self.mapped_reserve_size = 0
            self.entries = torch.empty((0, ENTRY_BYTES), dtype=torch.uint8)
            logger.warning(
                "MemFabric Engram no-GVA flow mode enabled: device=%d rank=%d/%d; "
                "native initialize/malloc/register_entry_table are all skipped",
                topology.device_id,
                topology.rank_id,
                topology.world_size,
            )
            return
        if self.gva_mode == "minimal":
            # acc_offload.initialize performs reserve, peer import and mmap;
            # malloc is only a suballocator. One GiB is HYBM's minimum aligned
            # SHARED slot and isolates the existence of peer GVA from table size.
            self.mapped_reserve_size = ONE_GIB
            self.allocation_size = ONE_GIB
        acl = _load_acl_context_api()
        caller_context = _get_acl_current_context(acl)
        _probe_tp_after_memfabric_init_stage(topology, "before-initialize", acl, caller_context)
        if _MEMFABRIC_TRACE:
            logger.warning(
                "[Engram MF trace] POOL initialization context saved "
                "device=%d mf_rank=%d context=%s",
                topology.device_id,
                topology.rank_id,
                "NULL" if caller_context is None else f"0x{caller_context:x}",
            )
        try:
            self.offload = _load_offload_module()
            cfg = self.offload.OffloadConfig()
            cfg.device_id = topology.device_id
            cfg.reserve_size = self.mapped_reserve_size
            cfg.alloc_size = self.allocation_size
            if self.diagnostic_local:
                # Diagnostic A/B against SHARED: retain the same DRAM size,
                # tensor views and model-loading path, but avoid the rank
                # group, peer import and shared-GVA mapping performed by the
                # SHARED scene. Runtime lookup is forbidden by the import-time
                # full-bypass guard above because entry_gather is SHARED-only.
                cfg.scene = self.offload.Scene.LOCAL
            else:
                if self.diagnostic_shared_single:
                    # Exercise the SHARED entity, store and mmap paths without
                    # importing a peer rank's slice/entity. Logical DP table
                    # ownership remains unchanged outside the MF config.
                    cfg.world_size = 1
                    cfg.rank_id = 0
                    cfg.local_world_size = 1
                else:
                    cfg.world_size = topology.world_size
                    cfg.rank_id = topology.rank_id
                    cfg.local_world_size = topology.local_world_size
                cfg.store_url = topology.store_url
                cfg.scene = self.offload.Scene.SHARED
            rc = self.offload.initialize(cfg)
            if rc:
                raise RuntimeError(f"offload.initialize failed: rc={rc}")
            self.initialized = True
            # HYBM may call aclrtSetDevice while constructing its GVA. Restore
            # the caller before touching torch_npu/HCCL in the diagnostic
            # probe, just as the normal constructor does on exit.
            _restore_acl_context(acl, caller_context)
            _probe_tp_after_memfabric_init_stage(topology, "after-initialize", acl, caller_context)
            if self.gva_mode == "minimal":
                self.entries = torch.empty((0, ENTRY_BYTES), dtype=torch.uint8)
                _restore_acl_context(acl, caller_context)
                _probe_tp_after_memfabric_init_stage(topology, "after-minimal-gva-ready", acl, caller_context)
            else:
                self.pointer = int(self.offload.malloc(self.allocation_size, 0))
                if not self.pointer:
                    raise RuntimeError(f"offload.malloc failed: size={self.allocation_size}")
                _restore_acl_context(acl, caller_context)
                _probe_tp_after_memfabric_init_stage(topology, "after-malloc", acl, caller_context)
                self._buffer = (ctypes.c_uint8 * local_grid_bytes).from_address(self.pointer)
                self.entries = torch.frombuffer(
                    self._buffer, dtype=torch.uint8, count=local_grid_bytes
                ).view(
                    layout.local_rows_span, ENTRY_BYTES
                )
                self.entries.zero_()
                if self.diagnostic_local:
                    _restore_acl_context(acl, caller_context)
                    _probe_tp_after_memfabric_init_stage(topology, "after-local-pool-ready", acl, caller_context)
                else:
                    rc = self.offload.register_entry_table(ENTRY_BYTES, layout.rows_per_slot)
                    if rc:
                        raise RuntimeError(f"offload.register_entry_table failed: rc={rc}")
                    _restore_acl_context(acl, caller_context)
                    _probe_tp_after_memfabric_init_stage(topology, "after-register-entry-table", acl, caller_context)
        except Exception as exc:
            try:
                self.close()
            except Exception as close_exc:  # noqa: BLE001 - retain both failures
                raise RuntimeError(f"{exc}; MemFabric rollback failed: {close_exc}") from exc
            raise RuntimeError(
                "MemFabric Engram pool initialization failed: "
                f"device={topology.device_id}, rank={topology.rank_id}/{topology.world_size}, "
                f"local_world={topology.local_world_size}, store={topology.store_url}, "
                f"reserve={self.reserve_size}, allocation={self.allocation_size}, "
                f"rows_per_slot={layout.rows_per_slot}: {exc}"
            ) from exc
        finally:
            observed_context, context_changed = _restore_acl_context(acl, caller_context)
            if _MEMFABRIC_TRACE:
                logger.warning(
                    "[Engram MF trace] POOL initialization context restored "
                    "device=%d mf_rank=%d saved=%s observed=%s changed=%s",
                    topology.device_id,
                    topology.rank_id,
                    "NULL" if caller_context is None else f"0x{caller_context:x}",
                    "NULL" if observed_context is None else f"0x{observed_context:x}",
                    context_changed,
                )

        logger.info(
            "Initialized MemFabric Engram pool: scene=%s scope=%s pool=%d tp_rank=%d "
            "rank=%d/%d local_world=%d device=%d "
            "store=%s rows_per_slot=%d local_rows_span=%d reserve=%.2f GiB "
            "allocation=%.2f GiB mapped=%.2f GiB",
            (
                "LOCAL_DIAGNOSTIC"
                if self.diagnostic_local
                else "SHARED_SINGLE_RANK_DIAGNOSTIC"
                if self.diagnostic_shared_single
                else "SHARED_REPLICATED_WORLD1"
                if self.replicated
                else "SHARED_MINIMAL_GVA"
                if self.gva_mode == "minimal"
                else "SHARED"
            ),
            topology.pool_scope,
            topology.pool_index,
            topology.tp_rank,
            topology.rank_id,
            topology.world_size,
            topology.local_world_size,
            topology.device_id,
            topology.store_url,
            layout.rows_per_slot,
            layout.local_rows_span,
            self.reserve_size / ONE_GIB,
            self.allocation_size / ONE_GIB,
            self.mapped_reserve_size / ONE_GIB,
        )
        if self.replicated:
            logger.warning(
                "MemFabric Engram is running in replicated world-size=1 mode: "
                "device=%d owns a complete table copy (%.2f GiB); DP peer "
                "sharing and peer GVA imports are disabled",
                topology.device_id,
                self.reserve_size / ONE_GIB,
            )

    @property
    def group(self):
        return get_tp_group() if self.topology.world_size > 1 else None

    def load_barrier(self, error: str | None) -> None:
        """Publish one layer's TP slots before any pool rank can use them."""
        group = self.group
        if group is None:
            if error is not None:
                raise RuntimeError(f"MemFabric Engram load failed: {error}")
            return
        errors: list[str | None] = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        if failures:
            raise RuntimeError(f"MemFabric Engram shared load failed: {failures}")

    def close(self) -> None:
        if self.closed:
            return
        self.entries = None
        self._buffer = None
        error = None
        if self.offload is not None and self.pointer:
            rc = self.offload.free(self.pointer, 0)
            if rc:
                error = RuntimeError(f"offload.free failed: rc={rc}, pointer=0x{self.pointer:x}")
            else:
                self.pointer = 0
        if self.offload is not None and self.initialized and error is None:
            rc = self.offload.uninitialize()
            if rc and error is None:
                error = RuntimeError(f"offload.uninitialize failed: rc={rc}")
            elif not rc:
                self.initialized = False
        if error is None:
            self.closed = True
            self.offload = None
            self.stream = None
            self.stream_handle = 0
            return
        raise error


class MemFabricEngramSession:
    """Model-level owner for one pool and every layer table view."""

    def __init__(self, layer_rows: list[tuple[int, int, int]], max_tokens: int, config) -> None:
        self.topology = MemFabricEngramTopology.from_runtime(config)
        local_layer_rows = [
            (layer_hash_index, rows) for layer_hash_index, rows, _ in layer_rows
        ]
        rank_layer_rows = None
        if self.topology.world_size > 1:
            group = get_tp_group()
            rank_layer_rows = [None] * self.topology.world_size
            dist.all_gather_object(
                rank_layer_rows,
                local_layer_rows,
                group=group.cpu_group,
            )
            if any(rows is None for rows in rank_layer_rows):
                raise RuntimeError("MemFabric Engram failed to gather TP-slot table layouts")
        self.layout = MemFabricEngramLayout.build(
            local_layer_rows,
            self.topology.world_size,
            rank_id=self.topology.rank_id,
            rank_layer_rows=rank_layer_rows,
        )
        self.pool = MemFabricEngramPool(self.topology, self.layout)
        self.views: dict[int, MemFabricEngramTableView] = {}
        self._shutdown_fenced = False
        self.closed = False
        device = torch.device("npu", self.topology.device_id)
        try:
            for layer_hash_index, _, local_heads in layer_rows:
                self.views[layer_hash_index] = MemFabricEngramTableView(
                    self.pool,
                    self.layout.segment(layer_hash_index),
                    max_tokens * local_heads,
                    device,
                    lookup_output_shape=(max_tokens, local_heads, CODE_BYTES),
                )
        except Exception:
            self.close(synchronize_group=False)
            raise

    def table(self, layer_hash_index: int) -> MemFabricEngramTableView:
        return self.views[layer_hash_index]

    def close(self, *, synchronize_group: bool = True) -> None:
        if self.closed:
            return
        if torch.npu.is_available():
            torch.npu.synchronize()
        group = self.pool.group
        if synchronize_group and group is not None and not self._shutdown_fenced:
            # No rank may unmap a slot while a peer can still issue a read.
            dist.barrier(group=group.cpu_group)
            self._shutdown_fenced = True
        for view in self.views.values():
            view.close()
        self.views.clear()
        self.pool.close()
        self.closed = True

    def __del__(self) -> None:
        # Deterministic runner shutdown is the primary path. This only covers
        # construction/load failures where the model was never published.
        with suppress(Exception):
            self.close(synchronize_group=False)
