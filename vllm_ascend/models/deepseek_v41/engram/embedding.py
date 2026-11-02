# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend head-sharded Engram tables with BF16, INT8 or native MXFP8 storage.

Subclasses upstream ``ParallelEngramEmbedding`` and preserves its parameter
and loader interface. Ascend supplies initialization, uniform head layout,
storage and lookup; upstream rank selection and hash gathering are reused:

* BF16 and A5 MXFP8 checkpoints retain their original rows and scales;
  INT8 checkpoints use group-32 FP32 scales;
* the table is either NPU memory or CANN-registered host memory read through a
  chunked device pointer table.
"""

import json
import os
import time
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import get_tp_group, in_the_same_node_as
from vllm.logger import logger
from vllm.model_executor.utils import set_weight_attrs

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

from .npu import (
    HostUvaBuffer,
    SharedUvaBuffer,
    gather_dequantize_engram_int8,
    gather_dequantize_host_uva,
    gather_dequantize_memfabric,
    quantize_engram_rows,
)
from .parallel import (
    engram_head_shard_rank,
    exchange_engram_rows,
    gather_engram_hashes,
    get_engram_dp_group,
    get_engram_dp_size,
)

_ENGRAM_MF_TRACE = os.getenv("VLLM_ASCEND_ENGRAM_MF_TRACE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_ENGRAM_MF_FRESH_OUTPUT = os.getenv("VLLM_ASCEND_ENGRAM_MF_FRESH_OUTPUT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_ENGRAM_MF_BYPASS = os.getenv("VLLM_ASCEND_ENGRAM_MF_BYPASS", "").strip().lower()
if _ENGRAM_MF_BYPASS not in ("", "full", "lookup"):
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_BYPASS must be one of '', 'full', or 'lookup', "
        f"got {_ENGRAM_MF_BYPASS!r}"
    )
_ENGRAM_MF_SKIP_LOAD = os.getenv("VLLM_ASCEND_ENGRAM_MF_SKIP_LOAD", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_ENGRAM_MF_BYPASS_FENCE = os.getenv("VLLM_ASCEND_ENGRAM_MF_BYPASS_FENCE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_ENGRAM_MF_TP_PROBE = os.getenv("VLLM_ASCEND_ENGRAM_MF_TP_PROBE", "").strip().lower()
if _ENGRAM_MF_TP_PROBE not in ("", "pre", "post", "both"):
    raise ValueError(
        "VLLM_ASCEND_ENGRAM_MF_TP_PROBE must be one of '', 'pre', 'post', or 'both', "
        f"got {_ENGRAM_MF_TP_PROBE!r}"
    )
try:
    _ENGRAM_MF_TRACE_CALLS = max(0, int(os.getenv("VLLM_ASCEND_ENGRAM_MF_TRACE_CALLS", "2")))
except ValueError:
    _ENGRAM_MF_TRACE_CALLS = 2
_engram_mf_embedding_trace_counts: dict[tuple[int, int], int] = {}
_engram_mf_bypass_counts: dict[tuple[int, int], int] = {}


def memfabric_full_bypass_enabled() -> bool:
    """Whether diagnostics must bypass the whole Engram prepare pipeline."""
    return _ENGRAM_MF_BYPASS == "full"


def memfabric_bypass_fence_enabled() -> bool:
    """Whether full-bypass output preparation is fenced for diagnostics."""
    return _ENGRAM_MF_BYPASS_FENCE


def _engram_mf_tp_probe(table, stage: str, tensor: torch.Tensor) -> None:
    """Fence only the TP input/output producer chain for MF diagnosis."""
    if table is None or _ENGRAM_MF_TP_PROBE not in (stage, "both") or tensor.numel() == 0:
        return
    topology = table.pool.topology
    logger.warning(
        "[Engram MF TP probe] stage=%s BEGIN device=%d mf_rank=%d layer=%d "
        "shape=%s stride=%s",
        stage,
        topology.device_id,
        table.rank_id,
        table.segment.layer_hash_index,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )
    started = time.perf_counter()
    value = tensor[0, 0, 0].item()
    logger.warning(
        "[Engram MF TP probe] stage=%s DONE device=%d mf_rank=%d layer=%d "
        "value=%s elapsed_ms=%.3f",
        stage,
        topology.device_id,
        table.rank_id,
        table.segment.layer_hash_index,
        value,
        (time.perf_counter() - started) * 1000,
    )


def _begin_engram_mf_embedding_trace(table, gathered: torch.Tensor) -> int | None:
    if not _ENGRAM_MF_TRACE or _ENGRAM_MF_TRACE_CALLS == 0 or table is None:
        return None
    key = (table.pool.topology.device_id, table.segment.layer_hash_index)
    previous_calls = _engram_mf_embedding_trace_counts.get(key, 0)
    if previous_calls >= _ENGRAM_MF_TRACE_CALLS:
        return None
    call = previous_calls + 1
    _engram_mf_embedding_trace_counts[key] = call
    logger.warning(
        "[Engram MF trace] EMBED BEGIN device=%d mf_rank=%d layer=%d call=%d "
        "gathered_shape=%s gathered_dtype=%s gathered_stride=%s",
        table.pool.topology.device_id,
        table.rank_id,
        table.segment.layer_hash_index,
        call,
        tuple(gathered.shape),
        gathered.dtype,
        tuple(gathered.stride()),
    )
    return call


class AscendParallelEngramEmbedding(ParallelEngramEmbedding):
    """TP(+EDP) head shard with BF16, INT8/FP32 or native MXFP8/E8M0."""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        layer_hash_index: int,
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        storage_dtype: torch.dtype = torch.int8,
        memfabric_table=None,
    ) -> None:
        self.storage_dtype = storage_dtype
        if storage_dtype not in (torch.bfloat16, torch.int8, torch.float8_e4m3fn):
            raise ValueError(f"Unsupported Engram storage dtype: {storage_dtype}")
        self.scale_dtype = torch.uint8 if storage_dtype == torch.float8_e4m3fn else torch.float32
        self.cpu_offload = cpu_offload
        self._memfabric_table = memfabric_table
        self.layer_hash_index = layer_hash_index
        self._shared_group = None
        if not all(in_the_same_node_as(get_tp_group().cpu_group)):
            raise ValueError("Ascend Engram requires the TP ranks of one replica to stay on a single node")
        group = get_engram_dp_group()
        if memfabric_table is None and group is not None and not all(in_the_same_node_as(group.cpu_group)):
            raise ValueError(
                "Ascend Engram requires all DP replicas to share the same node and shared-memory namespace"
            )
        if dp_shared_memory:
            if group is None or group.world_size <= 1:
                raise ValueError("dp_shared_memory needs a node-local sharing group with more than one rank")
            self._shared_group = group
            # Sharing replaces the per-step DP lookup collectives: every
            # replica looks up its own tokens over the mapped table.
            self.dp_size = 1
        else:
            self.dp_size = max(get_engram_dp_size(), 1)
        self._codes_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        self._scales_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        # The upstream constructor queries CUDA properties; keep its parameter
        # contract with Ascend storage and device allocation.
        nn.Module.__init__(self)
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        num_shards, head_rank = self._get_shard_info()
        if self.n_hash_cols % num_shards:
            raise ValueError(
                f"Engram requires uniform head shards: {self.n_hash_cols} heads "
                f"cannot be divided over {num_shards} TP x EDP shards. "
                "Use a divisible topology or enable dp_shared_memory."
            )
        self.part_n_hash_cols = self.n_hash_cols // num_shards
        self.head_start = head_rank * self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[: self.head_start + self.part_n_hash_cols])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False) if scales is not None else None
        for param in (self.weight, self.weight_scale_inv):
            if param is None:
                continue
            set_weight_attrs(param, {"weight_loader": self._weight_loader, "engram_vocab_start": self.vocab_start_idx})
        if cpu_offload:
            set_weight_attrs(self.weight, {"dummy_weight_value": 0})
            if self.weight_scale_inv is not None:
                scale_one = 127 if self.scale_dtype == torch.uint8 else 1.0
                set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": scale_one})
            table_bytes = (
                memfabric_table.local_entries.numel() * memfabric_table.local_entries.element_size()
                if memfabric_table is not None
                else weight.numel() * weight.element_size()
            )
            logger.info(
                "Engram table offloaded to %s: %d rows x %d, %.2f GiB per rank",
                "MemFabric DRAM (codes + scales)" if memfabric_table is not None else "registered host memory",
                self.part_num_embeddings,
                self.dim,
                (
                    table_bytes
                    if memfabric_table is not None
                    else table_bytes + (scales.numel() * scales.element_size() if scales is not None else 0)
                )
                / 1024**3,
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return self.tp_size, get_tensor_model_parallel_rank()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        codes_shape = (self.part_num_embeddings, self.dim)
        scales_shape = (self.part_num_embeddings, self.dim // self.block_size)
        device = torch.device("npu", torch.npu.current_device())
        quantized = self.storage_dtype != torch.bfloat16
        if getattr(self, "_memfabric_table", None) is not None:
            if not self.cpu_offload:
                raise ValueError("MemFabric Engram storage requires cpu_offload=True")
            if self.storage_dtype != torch.int8 or self.scale_dtype != torch.float32:
                raise ValueError("MemFabric Engram storage requires INT8 codes and FP32 scales")
            if self.dim != 256 or self.block_size != 32:
                raise ValueError("MemFabric Engram storage currently requires dim=256 and block_size=32")
            if self._memfabric_table.segment.rows != self.part_num_embeddings:
                raise ValueError(
                    "MemFabric Engram table layout mismatch: "
                    f"planned={self._memfabric_table.segment.rows}, embedding={self.part_num_embeddings}"
                )
            # The checkpoint iterator still needs named parameters carrying
            # the custom weight_loader callback. Both codes and scales live in
            # the model-owned pool, so these are zero-sized NPU sentinels.
            return (
                torch.empty(0, dtype=self.storage_dtype, device=device),
                torch.empty(0, dtype=self.scale_dtype, device=device),
            )
        if not self.cpu_offload:
            return (
                # Zeroed, not empty: the engine profiles the model (and runs the
                # Engram lookups) before the checkpoint is read, and a table of
                # uninitialised rows poisons the stream enough to break the MoE
                # routing later in that same forward.
                torch.zeros(codes_shape, dtype=self.storage_dtype, device=device),
                torch.zeros(scales_shape, dtype=self.scale_dtype, device=device) if quantized else None,
            )
        if self._shared_group is not None:
            # One physical copy of the mapped range, registered by each
            # rank in the sharing group.
            self._codes_uva = SharedUvaBuffer(codes_shape, self.storage_dtype, device, self._shared_group)
            if quantized:
                self._scales_uva = SharedUvaBuffer(scales_shape, self.scale_dtype, device, self._shared_group)
            return self._codes_uva.tensor, self._scales_uva.tensor if self._scales_uva is not None else None
        self._codes_uva = HostUvaBuffer(codes_shape, self.storage_dtype, device)
        if quantized:
            self._scales_uva = HostUvaBuffer(scales_shape, self.scale_dtype, device)
            self._scales_uva.tensor.zero_()
        # Same reason as the device path: aclrtMallocHost hands back whatever
        # was in the pages, and the profiling forward looks up before the
        # checkpoint load fills them.
        self._codes_uva.tensor.zero_()
        return self._codes_uva.tensor, self._scales_uva.tensor if self._scales_uva is not None else None

    def close_host_offload(self) -> None:
        """Release the registered host ranges (shutdown / reload path).

        The parameters alias the host range, so a released buffer has to take
        its alias with it: otherwise a later reader follows a live tensor into
        unmapped memory.  Each buffer and its own alias are dropped together,
        and only after that buffer's close succeeded, so a failed unregister
        stays retryable and a half-finished release never leaves a live
        parameter pointing at memory the other buffer has already freed.
        """
        aliases = {
            "_codes_uva": "weight",
            "_scales_uva": "weight_scale_inv",
        }
        for name, alias in aliases.items():
            buffer = getattr(self, name)
            if buffer is None:
                continue
            buffer.close()
            setattr(self, name, None)
            param = getattr(self, alias)
            setattr(
                self,
                alias,
                nn.Parameter(
                    torch.empty(0, dtype=param.dtype, device=param.device),
                    requires_grad=False,
                ),
            )

    def detach_memfabric(self) -> None:
        """Drop this embedding's non-owning table view before pool teardown."""
        self._memfabric_table = None

    def bind_checkpoint(self, model_path, key: str) -> None:
        """Use parameter callbacks while retaining the indexed shard reader.

        Codes and scales are read together from the selected index when the
        weight arrives. The scale callback is deliberately a no-op: iterator
        order and a separate unquantized index must not overwrite that pair.
        """
        self._checkpoint_path = model_path
        self._checkpoint_key = key

    def _weight_loader(self, param, loaded_weight) -> None:
        if param is self.weight:
            self.load_checkpoint(self._checkpoint_path, self._checkpoint_key)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Stream assigned rows, preserving BF16 in non-quantized models.

        Shared head slices have one writer per EDP group. The final CPU
        collective synchronizes writes and propagates loading failures.
        """
        if getattr(self, "_memfabric_table", None) is not None:
            if _ENGRAM_MF_SKIP_LOAD:
                # Diagnostic-only isolation of the SHARED pool lifecycle from
                # checkpoint I/O and slot writes. Every pool rank must still
                # enter the per-layer barrier so weight-loader ordering remains
                # identical to the normal path. Pair this with
                # VLLM_ASCEND_ENGRAM_MF_BYPASS=full: the table is deliberately
                # left uninitialized and must never be consumed by inference.
                logger.warning(
                    "[Engram MF diagnostic] skipping checkpoint load and slot writes for %s",
                    key,
                )
                self._memfabric_table.pool.load_barrier(None)
                return
            error = None
            try:
                self._load_into_storage(model_path, key, chunk_rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every pool rank
                error = f"{type(exc).__name__}: {exc}"
            self._memfabric_table.pool.load_barrier(error)
            return
        if self._shared_group is None:
            self._load_into_storage(model_path, key, chunk_rows)
            return
        error = None
        if self._shared_group.rank_in_group == 0:
            try:
                self._load_into_storage(model_path, key, chunk_rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every rank
                error = f"{type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * self._shared_group.world_size
        dist.all_gather_object(errors, error, group=self._shared_group.cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        if failures:
            raise RuntimeError(f"Engram shared load failed: {failures}")

    # The Ascend table is streamed out of these indexed safetensors shards.
    _INDEX_FILES = (
        "quant_model_weights.safetensors.index.json",
        "model.safetensors.index.json",
    )

    @classmethod
    def _checkpoint_index(cls, root: Path, key: str) -> dict[str, str]:
        """Weight map of the shard that holds ``key``.

        A checkpoint without an index, or a single-file/``pt`` one, cannot be
        served by this loader.  Inside the loader this is the last line of
        defence, after the tables are already allocated; the construction path
        runs `preflight_engram_checkpoint()` first so an unreadable checkpoint
        fails before any backing exists.
        """
        named = []
        for name in cls._INDEX_FILES:
            candidate = root / name
            if not candidate.is_file():
                continue
            named.append(name)
            weight_map = json.loads(candidate.read_text())["weight_map"]
            if key in weight_map:
                return weight_map
        raise ValueError(
            f"Engram table {key!r} is not in an indexed safetensors checkpoint "
            f"under {root} (found: {', '.join(named) or 'no index'})."
        )

    def _load_into_storage(self, model_path, key, chunk_rows) -> None:
        root = Path(model_path)
        scale_key = key.removesuffix(".weight") + ".scale"
        index = self._checkpoint_index(root, key)
        start, end = (self.vocab_start_idx, self.vocab_end_idx)
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            source_dtype = tensor.get_dtype()
            if getattr(self, "_memfabric_table", None) is not None:
                expected_shape = (self.num_embeddings, self.dim)
                if tuple(tensor.get_shape()) != expected_shape:
                    raise ValueError(
                        f"{key}: MemFabric Engram reload cannot change layout; "
                        f"expected {expected_shape}, got {tuple(tensor.get_shape())}. Restart the worker."
                    )
            if self.weight.dtype == torch.bfloat16:
                if tensor.get_dtype() != "BF16":
                    raise ValueError(f"{key}: expected BF16 source for BF16 storage")
                for chunk_start in range(start, end, chunk_rows):
                    stop = min(chunk_start + chunk_rows, end)
                    offset = chunk_start - start
                    self.weight.data[offset : offset + stop - chunk_start].copy_(tensor[chunk_start:stop])
            elif self.weight.dtype == torch.float8_e4m3fn:
                if source_dtype not in ("F8_E4M3", "F8_E4M3FN"):
                    raise ValueError(f"{key}: expected MXFP8 source for MXFP8 storage")
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    self._validate_memfabric_scale_shape(scale_key, scale)
                    if scale.get_dtype() != "F8_E8M0":
                        raise ValueError(f"{scale_key}: MXFP8 requires E8M0 scales")
                    for chunk_start in range(start, end, chunk_rows):
                        stop = min(chunk_start + chunk_rows, end)
                        offset = chunk_start - start
                        target_end = offset + stop - chunk_start
                        # Preserve the checkpoint bits; uint8 scales hold the
                        # E8M0 exponent, not a numerically cast integer value.
                        self._copy_codes(offset, tensor[chunk_start:stop].view(torch.uint8))
                        self.weight_scale_inv.data[offset:target_end].copy_(scale[chunk_start:stop].view(torch.uint8))
            elif source_dtype in ("I8", "INT8"):
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    self._validate_memfabric_scale_shape(scale_key, scale)
                    for chunk_start in range(start, end, chunk_rows):
                        stop = min(chunk_start + chunk_rows, end)
                        offset = chunk_start - self.vocab_start_idx
                        self._copy_quantized_rows(offset, tensor[chunk_start:stop], scale[chunk_start:stop])
            elif source_dtype in ("F8_E4M3", "F8_E4M3FN"):
                # Older devices use INT8/FP32 storage. Decode the checkpoint's
                # E8M0 group-32 scales before requantizing; A5 takes the native
                # bit-preserving load path above.
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    self._validate_memfabric_scale_shape(scale_key, scale)
                    for chunk_start in range(start, end, chunk_rows):
                        stop = min(chunk_start + chunk_rows, end)
                        offset = chunk_start - self.vocab_start_idx
                        values = tensor[chunk_start:stop].float()
                        group_scales = scale[chunk_start:stop].float()
                        values = (values.unflatten(-1, (-1, self.block_size)) * group_scales.unsqueeze(-1)).flatten(-2)
                        codes, scales = quantize_engram_rows(values)
                        self._copy_quantized_rows(offset, codes, scales)
            else:
                for chunk_start in range(start, end, chunk_rows):
                    stop = min(chunk_start + chunk_rows, end)
                    offset = chunk_start - self.vocab_start_idx
                    codes, scales = quantize_engram_rows(tensor[chunk_start:stop].to(torch.float32))
                    self._copy_quantized_rows(offset, codes, scales)
        logger.info("Engram rows [%d, %d) loaded from %s", start, end, index[key])

    def _copy_codes(self, local_row_start: int, codes: torch.Tensor) -> None:
        target_end = local_row_start + codes.shape[0]
        target = self.weight.data[local_row_start:target_end]
        if target.dtype == torch.float8_e4m3fn and codes.dtype == torch.uint8:
            target.view(torch.uint8).copy_(codes)
        else:
            target.copy_(codes)

    def _copy_quantized_rows(self, local_row_start: int, codes: torch.Tensor, scales: torch.Tensor) -> None:
        if getattr(self, "_memfabric_table", None) is not None:
            self._memfabric_table.copy_entries(local_row_start, codes.view(torch.int8), scales.float())
            return
        self._copy_codes(local_row_start, codes)
        target_end = local_row_start + scales.shape[0]
        self.weight_scale_inv.data[local_row_start:target_end].copy_(scales)

    def _validate_memfabric_scale_shape(self, key: str, scale) -> None:
        if getattr(self, "_memfabric_table", None) is None:
            return
        expected_shape = (self.num_embeddings, self.dim // self.block_size)
        if tuple(scale.get_shape()) != expected_shape:
            raise ValueError(
                f"{key}: MemFabric Engram reload cannot change scale layout; "
                f"expected {expected_shape}, got {tuple(scale.get_shape())}. Restart the worker."
            )

    def embed_gathered(self, gathered: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Embed ids already gathered across the EDP group.

        ``gathered`` is ``[slot * EDP, n_hash_cols]`` rank-major; each replica
        keeps its own ``num_tokens`` window. Returns ``[num_tokens, n_hash_cols,
        dim]`` bf16 with the heads back in checkpoint order.
        """
        bypass_lookup = self._memfabric_table is not None and _ENGRAM_MF_BYPASS == "lookup"
        if self._memfabric_table is not None and _ENGRAM_MF_BYPASS == "full":
            device_id = self._memfabric_table.pool.topology.device_id
            layer_hash_index = self._memfabric_table.segment.layer_hash_index
            bypass_key = (device_id, layer_hash_index)
            bypass_call = _engram_mf_bypass_counts.get(bypass_key, 0) + 1
            _engram_mf_bypass_counts[bypass_key] = bypass_call
            logger.warning(
                "[Engram MF bypass] device=%d mf_rank=%d layer=%d call=%d mode=full "
                "output_shape=(%d, %d, %d)",
                device_id,
                self._memfabric_table.rank_id,
                layer_hash_index,
                bypass_call,
                num_tokens,
                self.n_hash_cols,
                self.dim,
            )
            if _ENGRAM_MF_BYPASS_FENCE:
                logger.warning(
                    "[Engram MF bypass fence] device=%d mf_rank=%d layer=%d call=%d "
                    "stage=pre-allocation synchronize begin",
                    device_id,
                    self._memfabric_table.rank_id,
                    layer_hash_index,
                    bypass_call,
                )
                started = time.perf_counter()
                torch.npu.current_stream().synchronize()
                logger.warning(
                    "[Engram MF bypass fence] device=%d mf_rank=%d layer=%d call=%d "
                    "stage=pre-allocation synchronize returned elapsed_ms=%.3f",
                    device_id,
                    self._memfabric_table.rank_id,
                    layer_hash_index,
                    bypass_call,
                    (time.perf_counter() - started) * 1000,
                )
            out = torch.zeros(
                (num_tokens, self.n_hash_cols, self.dim),
                dtype=torch.bfloat16,
                device=gathered.device,
            )
            logger.warning(
                "[Engram MF bypass] device=%d mf_rank=%d layer=%d call=%d mode=full allocation returned "
                "output_shape=%s",
                device_id,
                self._memfabric_table.rank_id,
                layer_hash_index,
                bypass_call,
                tuple(out.shape),
            )
            if _ENGRAM_MF_BYPASS_FENCE:
                logger.warning(
                    "[Engram MF bypass fence] device=%d mf_rank=%d layer=%d call=%d "
                    "stage=post-allocation synchronize begin",
                    device_id,
                    self._memfabric_table.rank_id,
                    layer_hash_index,
                    bypass_call,
                )
                started = time.perf_counter()
                torch.npu.current_stream().synchronize()
                logger.warning(
                    "[Engram MF bypass fence] device=%d mf_rank=%d layer=%d call=%d "
                    "stage=post-allocation synchronize returned elapsed_ms=%.3f",
                    device_id,
                    self._memfabric_table.rank_id,
                    layer_hash_index,
                    bypass_call,
                    (time.perf_counter() - started) * 1000,
                )
            return out

        trace_call = _begin_engram_mf_embedding_trace(self._memfabric_table, gathered)
        # A layer slice of [tokens, layers, columns] has a non-unit token
        # stride but a unit column stride. Every lookup kernel accepts the
        # token stride explicitly, so keep this view and avoid an NPU
        # strided-to-contiguous copy for every Engram layer.
        # MemFabric owns a capacity-sized stable output scratch. Reusing it
        # avoids allocator/event dependencies in the forward path and gives
        # graph capture a fixed address. Other storage backends retain their
        # ordinary eager allocation.
        if bypass_lookup:
            out = torch.zeros(
                (gathered.shape[0], self.part_n_hash_cols, self.dim),
                dtype=torch.bfloat16,
                device=gathered.device,
            )
            logger.warning(
                "[Engram MF bypass] device=%d mf_rank=%d layer=%d mode=lookup "
                "local_output_shape=%s",
                self._memfabric_table.pool.topology.device_id,
                self._memfabric_table.rank_id,
                self._memfabric_table.segment.layer_hash_index,
                tuple(out.shape),
            )
        elif self._memfabric_table is not None:
            tokens = gathered.shape[0]
            if _ENGRAM_MF_FRESH_OUTPUT:
                # Diagnostic A/B against Host-UVA: bypass the model-owned,
                # stable lookup scratch without changing the MF lookup or TP
                # collective. The local variable keeps this allocation alive
                # until its consumer has been submitted.
                out = torch.empty(
                    (tokens, self.part_n_hash_cols, self.dim),
                    dtype=torch.bfloat16,
                    device=gathered.device,
                )
                logger.warning(
                    "[Engram MF diagnostic] device=%d mf_rank=%d layer=%d "
                    "using fresh lookup output shape=%s stride=%s",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    tuple(out.shape),
                    tuple(out.stride()),
                )
            else:
                scratch = self._memfabric_table.lookup_output
                capacity = 0 if scratch is None else scratch.shape[0]
                if trace_call is not None:
                    # Keep this runtime marker free of data_ptr(), allocator
                    # statistics and stream queries: those diagnostics can enter
                    # the NPU runtime themselves and obscure the actual boundary.
                    logger.warning(
                        "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                        "output-scratch selected tokens=%d capacity=%d full_reuse=%s",
                        self._memfabric_table.pool.topology.device_id,
                        self._memfabric_table.rank_id,
                        self._memfabric_table.segment.layer_hash_index,
                        trace_call,
                        tokens,
                        capacity,
                        tokens == capacity,
                    )
                if scratch is None or tokens > capacity:
                    raise ValueError(
                        f"MemFabric Engram lookup needs {tokens} token rows, "
                        f"but its stable output scratch holds {capacity}"
                    )
                # Prefill normally consumes the complete capacity. Preserve the
                # original tensor object in that case; decode uses a metadata-only
                # narrow view of the same stable storage.
                out = scratch if tokens == capacity else scratch.narrow(0, 0, tokens)
                if trace_call is not None:
                    logger.warning(
                        "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                        "output-view ready shape=%s stride=%s",
                        self._memfabric_table.pool.topology.device_id,
                        self._memfabric_table.rank_id,
                        self._memfabric_table.segment.layer_hash_index,
                        trace_call,
                        tuple(out.shape),
                        tuple(out.stride()),
                    )
        else:
            # lookup() covers every output element and explicitly writes zero
            # for IDs outside this shard, so a zero-fill is unnecessary.
            out = torch.empty(
                (gathered.shape[0], self.part_n_hash_cols, self.dim),
                dtype=torch.bfloat16,
                device=gathered.device,
            )
        if not bypass_lookup:
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d lookup call begin",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                )
            self.lookup(gathered, out)
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d lookup call returned",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                )
        if self.dp_size > 1:
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                    "dp-exchange call begin shape=%s stride=%s num_tokens=%d dp_size=%d",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                    tuple(out.shape),
                    tuple(out.stride()),
                    num_tokens,
                    self.dp_size,
                )
            out = exchange_engram_rows(out, num_tokens)
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                    "dp-exchange call returned shape=%s stride=%s",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                    tuple(out.shape),
                    tuple(out.stride()),
                )
        else:
            out = out[:num_tokens]
        if self.tp_size > 1:
            _engram_mf_tp_probe(self._memfabric_table, "pre", out)
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                    "tp-all-gather call begin shape=%s stride=%s tp_size=%d",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                    tuple(out.shape),
                    tuple(out.stride()),
                    self.tp_size,
                )
            out = tensor_model_parallel_all_gather(out, dim=1)
            _engram_mf_tp_probe(self._memfabric_table, "post", out)
            if trace_call is not None:
                logger.warning(
                    "[Engram MF trace] EMBED device=%d mf_rank=%d layer=%d call=%d "
                    "tp-all-gather call returned shape=%s stride=%s",
                    self._memfabric_table.pool.topology.device_id,
                    self._memfabric_table.rank_id,
                    self._memfabric_table.segment.layer_hash_index,
                    trace_call,
                    tuple(out.shape),
                    tuple(out.stride()),
                )
        return out[:, : self.n_hash_cols]

    @staticmethod
    def wait_lookup(done, external: bool = False) -> None:
        """Make the current stream wait for an asynchronous table lookup."""
        stream = torch.npu.current_stream()
        if external:
            done.wait(stream)
            done.reset(stream)
        else:
            stream.wait_event(done)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim].

        The shared mode has to be passed through: the replicas of one TP slot
        map the same table, so a gathered batch would make each of them embed
        EDP0's ids instead of its own window.
        """
        num_tokens = indices.shape[0]
        gathered = gather_engram_hashes(indices, dp_shared_memory=self._shared_group is not None)
        return self.embed_gathered(gathered, num_tokens)

    def lookup(self, indices: torch.Tensor, out: torch.Tensor, background: bool = False) -> None:
        """Look up this shard's heads into ``[tokens, padded_heads, dim]`` bf16.

        All local head columns are written, including zeroes for invalid IDs.
        """
        tokens = indices.shape[0]
        if tokens == 0 or self.part_n_hash_cols == 0:
            return
        launch = dict(
            head_start=self.head_start,
            local_heads=self.part_n_hash_cols,
            pad_heads=self.part_n_hash_cols,
            vocab_start=self.vocab_start_idx,
            vocab_end=self.vocab_end_idx,
            output=out.view(-1, self.dim),
        )
        if self._memfabric_table is not None:
            gather_dequantize_memfabric(
                self._memfabric_table,
                indices,
                **launch,
            )
        elif self._codes_uva is not None:
            codes_uva = cast(HostUvaBuffer, self._codes_uva)
            scales_uva = cast(HostUvaBuffer, self._scales_uva) if self._scales_uva is not None else None
            gather_dequantize_host_uva(codes_uva, scales_uva, indices, **launch)
        elif self.weight.device.type != "cpu":
            gather_dequantize_engram_int8(self.weight, self.weight_scale_inv, indices, self.dim, **launch)
        else:
            _torch_lookup(self, indices, out)


def _torch_lookup(embed: AscendParallelEngramEmbedding, indices, out) -> None:
    """CPU reference used by tests and bring-up without a device table."""
    heads = embed.part_n_hash_cols
    columns = indices[:, embed.head_start : embed.head_start + heads].long()
    owned = (columns >= embed.vocab_start_idx) & (columns < embed.vocab_end_idx)
    local = torch.where(owned, columns - embed.vocab_start_idx, 0).reshape(-1)
    # CPU index_select does not implement all float8 dtypes.
    if embed.weight.dtype == torch.float8_e4m3fn:
        codes = torch.index_select(embed.weight.data.view(torch.uint8), 0, local).view(embed.weight.dtype)
    else:
        codes = torch.index_select(embed.weight.data, 0, local)
    if embed.weight_scale_inv is None:
        decoded = codes
    else:
        scales = torch.index_select(embed.weight_scale_inv.data, 0, local)
        if scales.dtype == torch.uint8:
            scales = torch.where(
                scales == 255,
                float("nan"),
                torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127),
            )
        decoded = (codes.float().unflatten(-1, (-1, embed.block_size)) * scales.unsqueeze(-1)).flatten(-2).bfloat16()
    decoded = decoded.view(indices.shape[0], heads, embed.dim)
    out[:, :heads].copy_(torch.where(owned.unsqueeze(-1), decoded, 0.0))


def engram_storage_dtype(root, layer_id: int) -> torch.dtype:
    """Choose storage from the table header, independently of dense-layer quantization."""
    root = Path(root)
    key = f"layers.{layer_id}.engram.embed.weight"
    index = AscendParallelEngramEmbedding._checkpoint_index(root, key)
    with safe_open(root / index[key], framework="pt", device="cpu") as file:
        dtype = file.get_slice(key).get_dtype()
    if dtype == "BF16":
        return torch.bfloat16
    if dtype in ("I8", "INT8"):
        return torch.int8
    if dtype in ("F8_E4M3", "F8_E4M3FN"):
        return torch.float8_e4m3fn
    raise ValueError(f"{key}: unsupported Engram checkpoint dtype {dtype}")


def preflight_engram_checkpoint(root, layer_ids, embed_cls=AscendParallelEngramEmbedding) -> None:
    """Check that this checkpoint can be served *before* any table is allocated.

    The shard loader streams indexed safetensors, so a checkpoint without an
    index (or with a missing Engram key, or an INT8 weight without its scale)
    cannot be loaded here.  Finding that out during weight iteration would be
    after every rank has already allocated and registered its table,
    potentially the complete node table in row shared mode. Resolve the index
    entries and read the safetensors headers first.  The loader
    keeps its own check as the last line of defence.
    """
    root = Path(root)
    for layer_id in layer_ids:
        key = f"layers.{layer_id}.engram.embed.weight"
        index = embed_cls._checkpoint_index(root, key)
        shard = root / index[key]
        if not shard.is_file():
            raise ValueError(f"Engram layer {layer_id}: the checkpoint index points at {shard}, which does not exist.")
        with safe_open(shard, framework="pt", device="cpu") as file:
            source_dtype = file.get_slice(key).get_dtype()
        if source_dtype not in ("I8", "INT8", "F8_E4M3", "F8_E4M3FN"):
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        # The loader resolves scales from the weight's selected index too.
        if scale_key not in index:
            raise ValueError(
                f"Engram layer {layer_id}: {scale_key!r} is missing from the checkpoint index selected for {key!r}."
            )
        scale_shard = root / index[scale_key]
        if not scale_shard.is_file():
            raise ValueError(
                f"Engram layer {layer_id}: the checkpoint index points at "
                f"{scale_shard} for {scale_key!r}, which does not exist."
            )
        with safe_open(scale_shard, framework="pt", device="cpu") as file:
            if scale_key not in file.keys():  # noqa: SIM118 - safe_open is not a mapping
                raise ValueError(
                    f"Engram layer {layer_id}: {scale_shard} does not contain the scale tensor {scale_key!r}."
                )
            scale_dtype = file.get_slice(scale_key).get_dtype()
            if source_dtype in ("F8_E4M3", "F8_E4M3FN") and scale_dtype != "F8_E8M0":
                raise ValueError(f"Engram layer {layer_id}: MXFP8 table requires E8M0 scales, got {scale_dtype}.")
