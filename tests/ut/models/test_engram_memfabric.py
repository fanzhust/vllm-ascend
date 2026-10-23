# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Python contracts for model-owned MemFabric Engram storage."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram.memfabric import (
    MemFabricEngramLayout,
    MemFabricEngramTableView,
)


def test_memfabric_layout_folds_layers_into_one_uniform_grid():
    layout = MemFabricEngramLayout.build([(0, 5), (1, 8)], world_size=2)

    first = layout.segment(0)
    second = layout.segment(1)
    assert (first.capacity, first.layer_base) == (3, 0)
    assert (second.capacity, second.layer_base) == (4, 3)
    assert layout.rows_per_slot == 7

    # Logical row 4 of layer 0 belongs to rank 1, owner row 1.
    global_id = (4 // first.capacity) * layout.rows_per_slot + first.layer_base + 4 % first.capacity
    assert global_id == 8
    assert divmod(global_id, layout.rows_per_slot) == (1, 1)


def test_memfabric_layout_rejects_uint32_entry_id_overflow():
    with pytest.raises(ValueError, match="exceeds uint32"):
        MemFabricEngramLayout.build([(0, 1 << 32)], world_size=2)


def test_table_view_copies_only_its_owner_partition():
    view = object.__new__(MemFabricEngramTableView)
    view.owned_start = 3
    view.owned_end = 6
    view.local_codes = torch.zeros((3, 4), dtype=torch.int8)

    values = torch.arange(20, dtype=torch.int8).view(5, 4)
    view.copy_codes(local_row_start=1, codes=values)

    assert torch.equal(view.local_codes, values[2:5])


def test_zero_sized_loader_sentinel_populates_memfabric_codes_and_hbm_scales(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    codes = torch.arange(11 * 256, dtype=torch.int64).remainder(251).sub(125).to(torch.int8).view(11, 256)
    scales = torch.arange(11 * 8, dtype=torch.float32).view(11, 8).add_(1)
    save_file({key: codes, scale_key: scales}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )

    storage = torch.empty_like(codes)
    view = SimpleNamespace(copy_codes=MagicMock(), pool=SimpleNamespace(load_barrier=MagicMock()))
    view.copy_codes.side_effect = lambda start, chunk: storage[start : start + chunk.shape[0]].copy_(chunk)
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._memfabric_table = view
    table._shared_group = None
    table.num_embeddings = 11
    table.dim = 256
    table.block_size = 32
    table.vocab_start_idx = 0
    table.vocab_end_idx = 11
    table.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.zeros_like(scales), requires_grad=False)
    table.bind_checkpoint(tmp_path, key)

    table._weight_loader(table.weight, torch.empty(0))

    torch.testing.assert_close(storage, codes)
    torch.testing.assert_close(table.weight_scale_inv, scales)
    view.pool.load_barrier.assert_called_once_with(None)


def test_lookup_routes_memfabric_table_without_using_host_uva(monkeypatch):
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    table._memfabric_table = object()
    table._codes_uva = MagicMock()
    table.weight_scale_inv = torch.ones((5, 8))
    table.part_n_hash_cols = 1
    table.head_start = 0
    table.vocab_start_idx = 0
    table.vocab_end_idx = 5
    table.dim = 256
    memfabric = MagicMock()
    host_uva = MagicMock()
    monkeypatch.setattr(embedding_mod, "gather_dequantize_memfabric", memfabric)
    monkeypatch.setattr(embedding_mod, "gather_dequantize_host_uva", host_uva)
    ids = torch.tensor([[1], [2]], dtype=torch.int32)
    output = torch.empty((2, 1, 256), dtype=torch.bfloat16)

    table.lookup(ids, output)

    memfabric.assert_called_once()
    assert memfabric.call_args.args[0] is table._memfabric_table
    assert memfabric.call_args.args[1] is table.weight_scale_inv
    assert memfabric.call_args.args[2] is ids
    host_uva.assert_not_called()


def test_pool_close_keeps_failed_free_retryable():
    calls = []

    class FakeOffload:
        def __init__(self):
            self.fail_free = True

        def free(self, pointer, flags):
            calls.append(("free", pointer, flags))
            if self.fail_free:
                self.fail_free = False
                return -1
            return 0

        def uninitialize(self):
            calls.append(("uninitialize",))
            return 0

    from vllm_ascend.models.deepseek_v41.engram.memfabric import MemFabricEngramPool

    pool = object.__new__(MemFabricEngramPool)
    pool.offload = FakeOffload()
    pool.pointer = 1234
    pool.codes = torch.empty(0)
    pool._buffer = SimpleNamespace()
    pool.initialized = True
    pool.closed = False

    with pytest.raises(RuntimeError, match="offload.free failed"):
        pool.close()
    assert pool.initialized
    assert pool.pointer == 1234
    assert calls == [("free", 1234, 0)]

    pool.close()
    assert pool.closed
    assert pool.pointer == 0
    assert calls[-2:] == [("free", 1234, 0), ("uninitialize",)]
