# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side planning for MemFabric Engram head pools and row slots."""

from dataclasses import dataclass
from math import gcd

CODE_BYTES = 256
SCALE_COUNT = 8
SCALE_BYTES = SCALE_COUNT * 4
ENTRY_BYTES = CODE_BYTES + SCALE_BYTES
ONE_GIB = 1 << 30
MAX_ENTRY_ID = 1 << 32


@dataclass(frozen=True)
class EngramPoolMembership:
    pool_index: int
    rank_id: int
    members: tuple[int, ...]
    local_world_size: int


def plan_engram_pools(
    tp_workers: list[tuple[int, int]], workers_per_node: int
) -> tuple[int, list[EngramPoolMembership]]:
    """Return memberships indexed by machine-major deployment worker rank."""
    if workers_per_node < 1 or not tp_workers or len(tp_workers) % workers_per_node:
        raise ValueError("MemFabric Engram requires a fixed number of workers on every machine")
    if any(tp < 1 or not 0 <= rank < tp for tp, rank in tp_workers):
        raise ValueError("Invalid MemFabric Engram TP worker coordinates")
    common_tp = gcd(*(tp for tp, _ in tp_workers))
    nodes = len(tp_workers) // workers_per_node
    # A TP group can span machines. In that case, a common head boundary is
    # usable only if the resulting pool still has equal membership on every
    # machine. Try its coarser common boundaries under the same layout rules.
    for num_pools in range(common_tp, 0, -1):
        if common_tp % num_pools or workers_per_node % num_pools:
            continue
        groups: list[list[int]] = [[] for _ in range(num_pools)]
        for worker, (tp, rank) in enumerate(tp_workers):
            groups[rank // (tp // num_pools)].append(worker)
        local = workers_per_node // num_pools
        if any(
            len(group) != local * nodes
            or any(worker // workers_per_node != rank // local for rank, worker in enumerate(group))
            for group in groups
        ):
            continue
        memberships = {}
        for pool, group in enumerate(groups):
            members = tuple(group)
            for rank, worker in enumerate(group):
                memberships[worker] = EngramPoolMembership(pool, rank, members, local)
        return num_pools, [memberships[worker] for worker in range(len(tp_workers))]
    raise ValueError("MemFabric Engram cannot form uniform head pools")


@dataclass(frozen=True)
class MemFabricEngramSegment:
    layer_hash_index: int
    pool_row_base: int
    pool_rows: int
    layer_base: int
    owned_start: int
    rows: int


@dataclass(frozen=True)
class MemFabricEngramLayout:
    rows_per_slot: int
    local_rows_span: int
    segments: tuple[MemFabricEngramSegment, ...]

    @classmethod
    def build(
        cls,
        layer_heads: list[tuple[int, tuple[int, ...]]],
        num_pools: int,
        pool_index: int,
        world_size: int,
        rank_id: int,
    ) -> "MemFabricEngramLayout":
        if not 0 <= rank_id < world_size or not 0 <= pool_index < num_pools or not layer_heads:
            raise ValueError("Invalid MemFabric Engram pool layout")
        layer_base = 0
        segments = []
        for layer, heads in layer_heads:
            if not heads or len(heads) % num_pools or any(rows <= 0 for rows in heads):
                raise ValueError(f"Engram layer {layer} cannot be partitioned into {num_pools} head pools")
            heads_per_pool = len(heads) // num_pools
            first_head = pool_index * heads_per_pool
            row_base = sum(heads[:first_head])
            rows = sum(heads[first_head : first_head + heads_per_pool])
            base, extra = divmod(rows, world_size)
            if base == 0:
                raise ValueError(f"Engram layer {layer} has {rows} rows for {world_size} pool members")
            owned_start = row_base + rank_id * base + min(rank_id, extra)
            segments.append(
                MemFabricEngramSegment(layer, row_base, rows, layer_base, owned_start, base + (rank_id < extra))
            )
            layer_base += base + bool(extra)
        if layer_base >= MAX_ENTRY_ID or world_size * layer_base > MAX_ENTRY_ID:
            raise ValueError("MemFabric Engram entry grid exceeds uint32 IDs")
        local_rows_span = max(segment.layer_base + segment.rows for segment in segments)
        return cls(layer_base, local_rows_span, tuple(segments))
