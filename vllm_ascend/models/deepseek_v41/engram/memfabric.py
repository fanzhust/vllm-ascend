# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned MemFabric storage for DeepSeek-V4.1 Engram tables.

The process owns one SHARED pool participant. All Engram layers are folded
into one uniform 256-byte row grid because acc_offload exposes one registered
entry-table layout per process. Embeddings receive lightweight table views;
they never initialize or release the pool themselves.
"""

from __future__ import annotations

import ctypes
import socket
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger

from .parallel import get_engram_dp_group

ENTRY_BYTES = 256
ONE_GIB = 1 << 30
MAX_ENTRY_ID = 1 << 32


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
    segments: tuple[MemFabricEngramSegment, ...]

    @classmethod
    def build(cls, layer_rows: list[tuple[int, int]], world_size: int) -> MemFabricEngramLayout:
        if world_size <= 0:
            raise ValueError("MemFabric Engram world_size must be positive")
        layer_base = 0
        segments = []
        for layer_hash_index, rows in layer_rows:
            if rows <= 0:
                raise ValueError(f"Engram layer {layer_hash_index} must have at least one row")
            capacity = _ceil_div(rows, world_size)
            segments.append(MemFabricEngramSegment(layer_hash_index, rows, capacity, layer_base))
            layer_base += capacity
        if not segments:
            raise ValueError("MemFabric Engram needs at least one table")
        if world_size * layer_base >= MAX_ENTRY_ID:
            raise ValueError(
                f"MemFabric Engram entry grid exceeds uint32 IDs: world_size={world_size}, rows_per_slot={layer_base}"
            )
        return cls(world_size=world_size, rows_per_slot=layer_base, segments=tuple(segments))

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

    @classmethod
    def from_runtime(cls, config) -> MemFabricEngramTopology:
        group = get_engram_dp_group()
        if group is None:
            rank_id = 0
            world_size = 1
            local_world_size = 1
        else:
            rank_id = group.rank_in_group
            world_size = group.world_size
            hostnames: list[str | None] = [None] * world_size
            dist.all_gather_object(hostnames, socket.gethostname(), group=group.cpu_group)
            if any(host is None for host in hostnames):
                raise RuntimeError("MemFabric Engram failed to collect sharing-group host topology")
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
                            f"MemFabric Engram sharing-group ranks must be machine-major; got host order {hosts}"
                        )
                    seen.add(host)
                    previous = host
            local_world_size = counts[hosts[rank_id]]

        pool_index = get_tensor_model_parallel_rank()
        port = config.store_port_base + pool_index
        if port > 65535:
            raise ValueError(
                f"MemFabric Engram store port exceeds 65535: base={config.store_port_base}, pool_index={pool_index}"
            )
        store_url = config.store_url_template.format(port=port, pool_index=pool_index, tp_rank=pool_index)
        if len(store_url.encode()) >= 64:
            raise ValueError(f"MemFabric Engram store URL must be shorter than 64 bytes: {store_url!r}")
        return cls(
            rank_id=rank_id,
            world_size=world_size,
            local_world_size=local_world_size,
            device_id=torch.npu.current_device(),
            pool_index=pool_index,
            store_url=store_url,
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
    ) -> None:
        self.pool = pool
        self.segment = segment
        self.rank_id = pool.topology.rank_id
        self.rows_per_slot = pool.layout.rows_per_slot
        self.owned_start = self.rank_id * segment.capacity
        self.owned_end = min(self.owned_start + segment.capacity, segment.rows)
        owned_rows = self.owned_end - self.owned_start
        self.local_codes = pool.codes[segment.layer_base : segment.layer_base + owned_rows]
        self.global_ids = torch.empty(max_lookup_rows, dtype=torch.int64, device=device)
        self.count = torch.empty(1, dtype=torch.int32, device=device)
        self.staging = torch.empty((max_lookup_rows, ENTRY_BYTES), dtype=torch.int8, device=device)
        if self.staging.data_ptr() % ENTRY_BYTES:
            raise RuntimeError(
                f"MemFabric Engram staging buffer is not 256-byte aligned: address=0x{self.staging.data_ptr():x}"
            )
        self.closed = False

    @property
    def capacity(self) -> int:
        return self.segment.capacity

    @property
    def layer_base(self) -> int:
        return self.segment.layer_base

    def copy_codes(self, local_row_start: int, codes: torch.Tensor) -> None:
        """Copy the intersection of a logical TP-shard chunk owned by this rank."""
        local_row_end = local_row_start + codes.shape[0]
        start = max(local_row_start, self.owned_start)
        end = min(local_row_end, self.owned_end)
        if start >= end:
            return
        source_start = start - local_row_start
        target_start = start - self.owned_start
        self.local_codes[target_start : target_start + end - start].copy_(
            codes[source_start : source_start + end - start]
        )

    def close(self) -> None:
        if self.closed:
            return
        self.local_codes = None
        self.global_ids = None
        self.count = None
        self.staging = None
        self.pool = None
        self.closed = True


class MemFabricEngramPool:
    """The process-local participant in one sharing-group SHARED pool."""

    def __init__(self, topology: MemFabricEngramTopology, layout: MemFabricEngramLayout) -> None:
        self.topology = topology
        self.layout = layout
        self.offload: Any | None = None
        self.pointer = 0
        self.codes = None
        self._buffer = None
        self.initialized = False
        self.closed = False

        grid_bytes = layout.rows_per_slot * ENTRY_BYTES
        self.reserve_size = _align_up(grid_bytes, ONE_GIB)
        try:
            self.offload = _load_offload_module()
            cfg = self.offload.OffloadConfig()
            cfg.device_id = topology.device_id
            cfg.reserve_size = self.reserve_size
            cfg.alloc_size = self.reserve_size
            cfg.world_size = topology.world_size
            cfg.rank_id = topology.rank_id
            cfg.local_world_size = topology.local_world_size
            cfg.store_url = topology.store_url
            cfg.scene = self.offload.Scene.SHARED
            rc = self.offload.initialize(cfg)
            if rc:
                raise RuntimeError(f"offload.initialize failed: rc={rc}")
            self.initialized = True
            self.pointer = int(self.offload.malloc(self.reserve_size, 0))
            if not self.pointer:
                raise RuntimeError(f"offload.malloc failed: size={self.reserve_size}")
            self._buffer = (ctypes.c_int8 * grid_bytes).from_address(self.pointer)
            self.codes = torch.frombuffer(self._buffer, dtype=torch.int8, count=grid_bytes).view(
                layout.rows_per_slot, ENTRY_BYTES
            )
            self.codes.zero_()
            rc = self.offload.register_entry_table(ENTRY_BYTES, layout.rows_per_slot)
            if rc:
                raise RuntimeError(f"offload.register_entry_table failed: rc={rc}")
        except Exception as exc:
            try:
                self.close()
            except Exception as close_exc:  # noqa: BLE001 - retain both failures
                raise RuntimeError(f"{exc}; MemFabric rollback failed: {close_exc}") from exc
            raise RuntimeError(
                "MemFabric Engram pool initialization failed: "
                f"device={topology.device_id}, rank={topology.rank_id}/{topology.world_size}, "
                f"local_world={topology.local_world_size}, store={topology.store_url}, "
                f"reserve={self.reserve_size}, rows_per_slot={layout.rows_per_slot}: {exc}"
            ) from exc

        logger.info(
            "Initialized MemFabric Engram pool: rank=%d/%d local_world=%d device=%d "
            "store=%s rows_per_slot=%d reserve=%.2f GiB",
            topology.rank_id,
            topology.world_size,
            topology.local_world_size,
            topology.device_id,
            topology.store_url,
            layout.rows_per_slot,
            self.reserve_size / ONE_GIB,
        )

    def load_barrier(self, error: str | None) -> None:
        """Publish one layer's owner partition before any rank can use it."""
        group = get_engram_dp_group()
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
        self.codes = None
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
            return
        raise error


class MemFabricEngramSession:
    """Model-level owner for one pool and every layer table view."""

    def __init__(self, layer_rows: list[tuple[int, int, int]], max_tokens: int, config) -> None:
        self.topology = MemFabricEngramTopology.from_runtime(config)
        self.layout = MemFabricEngramLayout.build(
            [(layer_hash_index, rows) for layer_hash_index, rows, _ in layer_rows],
            self.topology.world_size,
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
        group = get_engram_dp_group()
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
