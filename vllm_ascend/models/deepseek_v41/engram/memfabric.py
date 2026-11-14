# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One acc_offload SHARED head pool per participating Engram worker."""

from __future__ import annotations

import ctypes
import json
from contextlib import suppress
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import torch
import torch.distributed as dist
from vllm.distributed import get_tensor_model_parallel_rank, get_world_group
from vllm.logger import logger

from vllm_ascend import envs

from .memfabric_layout import (
    CODE_BYTES,
    ENTRY_BYTES,
    ONE_GIB,
    SCALE_BYTES,
    SCALE_COUNT,
    EngramPoolMembership,
    MemFabricEngramLayout,
    MemFabricEngramSegment,
    plan_engram_pools,
)

_STARTUP_TIMEOUT = timedelta(minutes=30)
_SHUTDOWN_TIMEOUT = timedelta(seconds=60)


class EngramPoolControl:
    """Deployment-wide CPU rendezvous shared by independent P/D engines."""

    def __init__(self, config, parallel):
        if config.num_nodes < parallel.nnodes:
            raise ValueError("engram_memfabric.num_nodes must cover all vLLM worker machines")
        self.tp = parallel.tensor_parallel_size
        self.tp_rank = get_tensor_model_parallel_rank()
        world = get_world_group()
        if world.world_size != parallel.data_parallel_size * self.tp:
            raise ValueError("MemFabric Engram requires a vLLM world group containing the instance's DP × TP workers")
        # The world group covers this vLLM instance, including DP ranks
        # started by separate launchers. Each P/D instance discovers its own
        # placement before joining the shared MemFabric control store.
        node_ranks = [-1] * world.world_size
        dist.all_gather_object(node_ranks, config.node_rank, group=world.cpu_group)
        if any(not 0 <= node < config.num_nodes for node in node_ranks):
            raise ValueError(f"Invalid MemFabric Engram machine coordinates in vLLM world group: {node_ranks}")
        local_workers = [rank for rank, node in enumerate(node_ranks) if node == config.node_rank]
        local_rank = local_workers.index(world.rank_in_group)
        self.local_size = len(local_workers)
        self.world_size = self.local_size * config.num_nodes
        self.rank = config.node_rank * self.local_size + local_rank
        self.num_nodes = config.num_nodes
        self.node_rank = config.node_rank
        self.local_rank = local_rank
        endpoint = urlsplit(config.store_url)
        self.host = endpoint.hostname
        self.pool_port = endpoint.port
        self.control_port = config.control_port
        self.store = dist.TCPStore(
            self.host,
            self.control_port,
            world_size=None,
            is_master=self.rank == 0,
            timeout=_STARTUP_TIMEOUT,
            wait_for_workers=False,
            use_libuv=False,
        )

    def publish(self, stage: str, value) -> None:
        key = f"{stage}/{self.rank}"
        self.store.set(key, json.dumps(value))

    def collect(self, stage: str, members=None, *, timeout=_STARTUP_TIMEOUT) -> list:
        if members is None:
            members = range(self.world_size)
        keys = [f"{stage}/{rank}" for rank in members]
        try:
            self.store.wait(keys, timeout)
            return [json.loads(value) for value in self.store.multi_get(keys)]
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(f"MemFabric Engram rendezvous failed at {stage} on worker {self.rank}: {exc}") from exc

    def exchange_errors(self, stage: str, error: str | None) -> None:
        self.publish(stage, error)
        failures = [(rank, value) for rank, value in enumerate(self.collect(stage)) if value is not None]
        if failures:
            raise RuntimeError(f"MemFabric Engram {stage} failed: {failures}")

    def plan(self, identity: str | None, preflight_error: str | None) -> tuple[int, EngramPoolMembership]:
        coordinate = f"{self.node_rank}/{self.local_rank}"
        claim = uuid4().hex
        if self.store.compare_set(f"claim/{coordinate}", "", claim) != claim.encode():
            raise ValueError(f"Duplicate MemFabric Engram worker coordinate {coordinate}")
        # Publish with SET so TCPStore wakes ranks already waiting on this key.
        self.store.set(f"worker/{coordinate}", json.dumps([self.tp, self.tp_rank, identity, preflight_error]))
        expected_node = [self.local_size, self.num_nodes, self.host, self.pool_port]
        if self.local_rank == 0:
            self.store.set(f"node/{self.node_rank}", json.dumps(expected_node))
        nodes = [f"node/{node}" for node in range(self.num_nodes)]
        self.store.wait(nodes, _STARTUP_TIMEOUT)
        node_settings = [json.loads(value) for value in self.store.multi_get(nodes)]
        if any(settings != expected_node for settings in node_settings):
            raise ValueError(f"MemFabric Engram nodes disagree on worker count or rendezvous settings: {node_settings}")

        workers = [f"worker/{node}/{local}" for node in range(self.num_nodes) for local in range(self.local_size)]
        self.store.wait(workers, _STARTUP_TIMEOUT)
        metadata = [json.loads(value) for value in self.store.multi_get(workers)]
        failures = [(rank, data[3]) for rank, data in enumerate(metadata) if data[3] is not None]
        if failures:
            raise RuntimeError(f"MemFabric Engram checkpoint preflight failed: {failures}")
        if any(data[2] != identity for data in metadata):
            raise ValueError("MemFabric Engram workers have different checkpoint identities or head layouts")
        num_pools, memberships = plan_engram_pools([(data[0], data[1]) for data in metadata], self.local_size)
        if self.pool_port + num_pools > 65536 or self.pool_port <= self.control_port < self.pool_port + num_pools:
            raise ValueError("MemFabric Engram pool ports must fit in the TCP port range and exclude control_port")
        return num_pools, memberships[self.rank]

    def store_url(self, pool_index: int) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        url = f"tcp://{host}:{self.pool_port + pool_index}"
        if len(url.encode()) >= 64:
            raise ValueError(f"MemFabric Engram store URL must be shorter than 64 bytes: {url}")
        return url


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


class MemFabricEngramTableView:
    """A layer's local backing and stable NPU lookup buffers."""

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
        self.max_lookup_rows = max_lookup_rows
        self.fused_lookup_desc = None
        self.rows_per_slot = pool.layout.rows_per_slot
        self.code_bytes = CODE_BYTES
        self.owner_rows, self.extra_owners = divmod(segment.pool_rows, len(pool.membership.members))
        self.safe_entry_id = pool.membership.rank_id * self.rows_per_slot + segment.layer_base
        self.local_entries = pool.entries[segment.layer_base : segment.layer_base + segment.rows]
        self.global_ids = self.count = self._staging_storage = self.staging = self.staging_fp32 = None
        if not pool.use_fused_lookup:
            self.global_ids = torch.empty(max_lookup_rows, dtype=torch.int64, device=device)
            self.count = torch.empty(1, dtype=torch.int32, device=device)
            staging_bytes = max_lookup_rows * ENTRY_BYTES
            self._staging_storage = torch.empty(staging_bytes + ENTRY_BYTES - 1, dtype=torch.int8, device=device)
            staging_offset = (-self._staging_storage.data_ptr()) % ENTRY_BYTES
            self.staging = self._staging_storage[staging_offset : staging_offset + staging_bytes].view(
                max_lookup_rows, ENTRY_BYTES
            )
            self.staging_fp32 = self.staging.view(torch.float32)
        self.lookup_output = (
            torch.empty(lookup_output_shape, dtype=torch.bfloat16, device=device)
            if lookup_output_shape is not None
            else None
        )
        self.closed = False

    def configure_fused_lookup(self, head_start: int, local_heads: int, vocab_start: int, vocab_end: int) -> None:
        if self.pool.use_fused_lookup:
            self.fused_lookup_desc = self.pool.offload.EngramLookupDesc(
                head_start=head_start,
                local_heads=local_heads,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
                pool_row_base=self.segment.pool_row_base,
                owner_rows=self.owner_rows,
                extra_owners=self.extra_owners,
                layer_base=self.segment.layer_base,
            )

    def copy_entries(self, local_row_start: int, codes: torch.Tensor, scales: torch.Tensor) -> None:
        """Copy locally owned rows as packed [256B codes, 32B scales]."""
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
                "MemFabric Engram local row write is outside the owned slot segment: "
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
        self.fused_lookup_desc = None
        self.global_ids = None
        self.count = None
        self.staging_fp32 = None
        self.staging = None
        self._staging_storage = None
        self.lookup_output = None
        self.pool = None
        self.closed = True


class MemFabricEngramPool:
    """A process's sole acc_offload session."""

    def __init__(
        self,
        device_id: int,
        membership: EngramPoolMembership,
        store_url: str,
        layout: MemFabricEngramLayout,
        *,
        use_fused_lookup: bool = False,
    ) -> None:
        self.layout = layout
        self.offload: Any = None
        self.pointer = 0
        self.initialized = False
        self.ready = False
        self.entries = None
        self._buffer = None
        self.device_id = device_id
        self.membership = membership
        self.use_fused_lookup = use_fused_lookup
        slot_size = ((layout.rows_per_slot * ENTRY_BYTES + ONE_GIB - 1) // ONE_GIB) * ONE_GIB
        backed_bytes = layout.local_rows_span * ENTRY_BYTES
        acl = _load_acl_context_api()
        caller_context = _get_acl_current_context(acl)
        try:
            from memfabric_hybrid import offload

            self.offload = offload
            cfg = offload.OffloadConfig()
            if not (
                hasattr(cfg, "local_world_size")
                and hasattr(cfg, "store_url")
                and callable(getattr(offload, "register_entry_table", None))
                and callable(getattr(offload, "entry_gather", None))
            ):
                raise RuntimeError("Install a MemFabric wheel with multi-node SHARED and entry_gather support")
            cfg.device_id = device_id
            cfg.reserve_size = slot_size
            cfg.alloc_size = slot_size
            cfg.world_size = len(membership.members)
            cfg.rank_id = membership.rank_id
            cfg.local_world_size = membership.local_world_size
            cfg.store_url = store_url
            cfg.scene = offload.Scene.SHARED
            if rc := offload.initialize(cfg):
                raise RuntimeError(f"offload.initialize failed: {rc}")
            self.initialized = True
            _restore_acl_context(acl, caller_context)
            self.pointer = int(offload.malloc(slot_size, 0))
            if not self.pointer:
                raise RuntimeError("offload.malloc failed")
            _restore_acl_context(acl, caller_context)
            if rc := offload.register_entry_table(ENTRY_BYTES, layout.rows_per_slot):
                raise RuntimeError(f"offload.register_entry_table failed: {rc}")
            _restore_acl_context(acl, caller_context)
            self._buffer = (ctypes.c_uint8 * backed_bytes).from_address(self.pointer)
            self.entries = torch.frombuffer(self._buffer, dtype=torch.uint8).view(-1, ENTRY_BYTES)
            self.entries.zero_()
        except BaseException:
            self.close()
            raise
        finally:
            _restore_acl_context(acl, caller_context)
        logger.info(
            "MemFabric Engram pool=%d rank=%d/%d local_world=%d device=%d store=%s rows_per_slot=%d slot_bytes=%d "
            "lookup=%s",
            membership.pool_index,
            membership.rank_id,
            len(membership.members),
            membership.local_world_size,
            device_id,
            store_url,
            layout.rows_per_slot,
            slot_size,
            "fused" if use_fused_lookup else "legacy",
        )

    def close(self) -> None:
        self.ready = False
        self.entries = None
        self._buffer = None
        if self.pointer:
            self.offload.free(self.pointer, 0)
            self.pointer = 0
        if self.initialized:
            self.offload.uninitialize()
            self.initialized = False


class MemFabricEngramSession:
    """Own one head pool and its views for the lifetime of the model."""

    def __init__(
        self,
        layer_heads: list[tuple[int, tuple[int, ...]]],
        max_tokens: int,
        config,
        parallel,
        checkpoint_identity,
    ) -> None:
        self.control = EngramPoolControl(config, parallel)
        self.views: dict[int, MemFabricEngramTableView] = {}
        self.pool = None
        identity = preflight_error = None
        try:
            identity = checkpoint_identity()
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
        num_pools, membership = self.control.plan(identity, preflight_error)
        layout = None
        use_fused_lookup = False
        error = None
        try:
            use_fused_lookup = envs.VLLM_ASCEND_ENGRAM_MF_FUSED
            if use_fused_lookup:
                from memfabric_hybrid import offload

                if not (
                    callable(getattr(offload, "engram_entry_gather", None))
                    and callable(getattr(offload, "EngramLookupDesc", None))
                ):
                    raise RuntimeError(
                        "Fused MemFabric Engram lookup requires a wheel with engram_entry_gather support"
                    )
            if any(len(heads) % self.control.tp for _, heads in layer_heads):
                raise ValueError(f"Engram heads cannot be divided over TP{self.control.tp}")
            layout = MemFabricEngramLayout.build(
                layer_heads, num_pools, membership.pool_index, len(membership.members), membership.rank_id
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.control.exchange_errors("layout", error)
        assert layout is not None
        device_id = torch.npu.current_device()
        device = torch.device("npu", device_id)
        error = None
        try:
            self.pool = MemFabricEngramPool(
                device_id,
                membership,
                self.control.store_url(membership.pool_index),
                layout,
                use_fused_lookup=use_fused_lookup,
            )
            for segment, (_, heads) in zip(layout.segments, layer_heads, strict=True):
                local_heads = len(heads) // self.control.tp
                self.views[segment.layer_hash_index] = MemFabricEngramTableView(
                    self.pool,
                    segment,
                    max_tokens * local_heads,
                    device,
                    lookup_output_shape=(max_tokens, local_heads, CODE_BYTES),
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            self.control.exchange_errors("mapping", error)
        except BaseException:
            self.close(synchronize_group=False)
            raise

    def table(self, layer_hash_index: int) -> MemFabricEngramTableView:
        return self.views[layer_hash_index]

    def finish_loading(self, error: str | None) -> None:
        self.control.exchange_errors("checkpoint load", error)
        self.pool.ready = True
        if self.control.rank == 0:
            logger.info(
                "MemFabric Engram ready on %d workers across %d machines",
                self.control.world_size,
                self.control.num_nodes,
            )

    def close(self, *, synchronize_group: bool = True) -> None:
        pool = self.pool
        if pool is None:
            return
        membership = pool.membership
        coordinated = synchronize_group and pool.ready
        if coordinated:
            torch.npu.synchronize(pool.device_id)
            self.control.publish("drained", True)
            self.control.collect("drained", timeout=_SHUTDOWN_TIMEOUT)
            # Each native pool's store remains alive while its peers unmap.
            if membership.rank_id == 0:
                peers = membership.members[1:]
                if peers:
                    self.control.collect("released", peers, timeout=_SHUTDOWN_TIMEOUT)
        for view in self.views.values():
            view.close()
        self.views.clear()
        pool.close()
        self.pool = None
        if coordinated:
            self.control.publish("released", True)
            if self.control.rank == 0:
                self.control.collect("released", timeout=_SHUTDOWN_TIMEOUT)

    def __del__(self) -> None:
        # A live deployment must drain every reader before releasing its rows.
        with suppress(Exception):
            if self.pool is not None and not self.pool.ready:
                self.close(synchronize_group=False)
