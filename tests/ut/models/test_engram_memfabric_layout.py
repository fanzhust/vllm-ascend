# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU-side head pool planner."""

import pytest

from vllm_ascend.models.deepseek_v41.engram.memfabric_layout import MemFabricEngramLayout, plan_engram_pools


@pytest.mark.parametrize(
    "topology,num_pools,local_size",
    [([(2, 8)], 8, 2), ([(4, 4), (8, 2)], 2, 8), ([(4, 4)] * 3 + [(8, 2)], 2, 8)],
)
def test_pool_members_are_machine_major_and_cover_all_worker_heads(topology, num_pools, local_size):
    workers = [(tp, rank) for dp, tp in topology for _ in range(dp) for rank in range(tp)]
    count, plan = plan_engram_pools(workers, 16)
    assert count == num_pools
    for worker, ((tp, rank), membership) in enumerate(zip(workers, plan, strict=True)):
        assert membership.pool_index == rank // (tp // count)
        assert membership.local_world_size == local_size
        assert len(membership.members) == local_size * len(topology)
        assert membership.members[membership.rank_id] == worker
        assert membership.rank_id // local_size == worker // 16


def test_unequal_physical_placement_uses_a_coarser_common_boundary():
    count, memberships = plan_engram_pools([(4, rank) for rank in range(4)], 2)
    assert count == 1
    assert [member.rank_id for member in memberships] == list(range(4))


def test_layer_slots_cover_each_pool_row_once_even_when_heads_have_unequal_sizes():
    heads = [(0, (3, 5, 7, 9)), (1, (7, 2, 4, 3))]
    layouts = [MemFabricEngramLayout.build(heads, 2, 1, 3, rank) for rank in range(3)]
    assert [layout.rows_per_slot for layout in layouts] == [9] * 3
    for layer, row_base, rows, layer_base in ((0, 8, 16, 0), (1, 9, 7, 6)):
        segments = [layout.segments[layer] for layout in layouts]
        assert {(s.pool_row_base, s.pool_rows, s.layer_base) for s in segments} == {(row_base, rows, layer_base)}
        covered = [row for s in segments for row in range(s.owned_start, s.owned_start + s.rows)]
        assert covered == list(range(row_base, row_base + rows))


def test_pool_rejects_an_unbacked_owner_and_uint32_overflow():
    with pytest.raises(ValueError, match="pool members"):
        MemFabricEngramLayout.build([(0, (2,))], 1, 0, 3, 0)
    with pytest.raises(ValueError, match="uint32"):
        MemFabricEngramLayout.build([(0, ((1 << 32) + 1,))], 1, 0, 2, 0)
