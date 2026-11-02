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

from vllm_ascend.model_executor.warmup import engram_memfabric_warmup as ordered_warmup
from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import memfabric as memfabric_mod
from vllm_ascend.models.deepseek_v41.engram import npu as npu_mod
from vllm_ascend.models.deepseek_v41.engram.memfabric import (
    CODE_BYTES,
    ENTRY_BYTES,
    SCALE_BYTES,
    SCALE_COUNT,
    MemFabricEngramLayout,
    MemFabricEngramPool,
    MemFabricEngramTableView,
    MemFabricEngramTopology,
)


def _early_init_config(*, dp_shared_memory=True):
    model_config = SimpleNamespace(hf_config=object(), model="unused", dtype=torch.bfloat16)
    return SimpleNamespace(
        model_config=model_config,
        quant_config=object(),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        engram_config=SimpleNamespace(dp_shared_memory=dp_shared_memory),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8192),
        load_config=SimpleNamespace(load_format="dummy"),
    )


def _patch_early_init_dependencies(monkeypatch, events):
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        index_topk=1,
        engram_layer_ids=(1,),
        engram_num_embeddings=(20,),
        engram_head_dim=256,
    )
    layout = SimpleNamespace(primes=(((10,) * 8,),))
    ascend_config = SimpleNamespace(
        engram_offload_backend="memfabric",
        engram_memfabric=object(),
        multistream_engram_overlap=False,
    )
    monkeypatch.setattr(model_mod, "normalize_deepseek_v41_config", lambda _: config)
    monkeypatch.setattr(model_mod, "_engram_enabled_for_runtime", lambda *_: True)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(model_mod, "engram_cpu_offload", lambda _: True)
    monkeypatch.setattr(model_mod, "resolve_dp_shared_memory", lambda requested: requested)
    monkeypatch.setattr(model_mod.EngramLayout, "from_config", lambda _: layout)
    monkeypatch.setattr(
        model_mod,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _: False),
    )
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(model_mod, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(model_mod, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeSession:
        def __init__(self, layer_rows, max_tokens, memfabric_config):
            events.append(("session", layer_rows, max_tokens, memfabric_config))

    monkeypatch.setattr(model_mod, "MemFabricEngramSession", FakeSession)
    return config, ascend_config


def test_memfabric_gva_is_initialized_before_first_model_npu_allocation(monkeypatch):
    events = []
    _, ascend_config = _patch_early_init_dependencies(monkeypatch, events)

    def stop_at_first_model_allocation(*args, **kwargs):
        events.append(("tensor-allocation", args, kwargs))
        raise RuntimeError("stop after observing the first model allocation")

    monkeypatch.setattr(model_mod.torch, "empty", stop_at_first_model_allocation)

    with pytest.raises(RuntimeError, match="stop after observing"):
        model_mod.DeepseekV41Model(vllm_config=_early_init_config())

    assert [event[0] for event in events] == ["session", "tensor-allocation"]
    assert events[0][1] == [(0, 10, 1)]
    assert events[0][2] == 8192
    assert events[0][3] is ascend_config.engram_memfabric


def test_memfabric_multi_dp_requires_shared_memory_semantics(monkeypatch):
    events = []
    _patch_early_init_dependencies(monkeypatch, events)

    with pytest.raises(ValueError, match="requires dp_shared_memory=true"):
        model_mod.DeepseekV41Model(vllm_config=_early_init_config(dp_shared_memory=False))

    assert events == []


def test_memfabric_full_bypass_skips_hash_pipeline_and_zeros_fixed_buffers(monkeypatch):
    table = SimpleNamespace(n_hash_cols=24, dim=8)
    hash_state = SimpleNamespace(ensure_cache=MagicMock(side_effect=AssertionError("hash pipeline must be skipped")))
    model = SimpleNamespace(
        has_engram=True,
        config=SimpleNamespace(engram_layer_ids=(1,)),
        layers={1: SimpleNamespace(engram=SimpleNamespace(embed_tokens=table))},
        engram_hash=hash_state,
        _engram_offload_session=SimpleNamespace(topology=SimpleNamespace(device_id=6, rank_id=0)),
    )
    model._prepare_engram_memfabric_full_bypass = (
        lambda *args, **kwargs: model_mod.DeepseekV41Model._prepare_engram_memfabric_full_bypass(
            model, *args, **kwargs
        )
    )
    monkeypatch.setattr(model_mod, "memfabric_full_bypass_enabled", lambda: True)
    monkeypatch.setattr(model_mod, "memfabric_bypass_fence_enabled", lambda: False)

    output = torch.full((7, table.n_hash_cols * table.dim), float("nan"), dtype=torch.bfloat16)
    mask = torch.ones(7, dtype=torch.bool)
    lookups, returned_mask = model_mod.DeepseekV41Model.prepare_engram(
        model,
        torch.arange(3, dtype=torch.int32),
        torch.arange(3, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        output_buffers={1: output},
        output_tokens=5,
        mask_output_buffer=mask,
    )

    hash_state.ensure_cache.assert_not_called()
    assert lookups[1] is output
    assert returned_mask is mask
    assert torch.count_nonzero(output[:5]) == 0
    assert torch.isnan(output[5:]).all()
    assert torch.count_nonzero(mask[:5]) == 0
    assert mask[5:].all()


def test_memfabric_entry_contains_codes_and_fp32_scales():
    assert CODE_BYTES == 256
    assert SCALE_COUNT == 8
    assert SCALE_BYTES == 32
    assert ENTRY_BYTES == 288


def test_ordered_warmup_materializes_hccl_before_aicpu(monkeypatch):
    events = []

    class Backend:
        def get_hccl_comm_name(self, rank):
            events.append(("communicator", rank))
            return "comm"

    class Group:
        def _get_backend(self, device):
            events.append(("backend", device.type))
            return Backend()

    coordinator = SimpleNamespace(device_group=Group())
    vllm_config = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(engram_layer_ids=(1,))),
    )
    monkeypatch.setattr(
        ordered_warmup,
        "get_ascend_config",
        lambda: SimpleNamespace(engram_offload_backend="memfabric"),
    )
    monkeypatch.setattr(ordered_warmup, "enable_custom_op", lambda: True)
    monkeypatch.setattr(ordered_warmup, "get_mc2_group", lambda: coordinator)
    monkeypatch.setattr(ordered_warmup, "get_ep_group", lambda: coordinator)
    monkeypatch.setattr(ordered_warmup.torch.distributed, "get_rank", lambda *, group: 0)
    monkeypatch.setattr(
        ordered_warmup,
        "_warm_up_sparse_flash_mla_metadata",
        lambda _: events.append(("aicpu",)),
    )

    assert ordered_warmup.warm_up_before_memfabric_engram(vllm_config, torch.device("cpu"))
    assert events == [
        ("backend", "npu"),
        ("communicator", 0),
        ("backend", "npu"),
        ("communicator", 0),
        ("aicpu",),
    ]


def test_memfabric_runner_fence_is_limited_to_full_bypass_diagnostics(monkeypatch):
    stream = MagicMock()
    model = SimpleNamespace(
        _engram_offload_session=SimpleNamespace(topology=SimpleNamespace(device_id=6, rank_id=0))
    )
    monkeypatch.setattr(model_mod, "memfabric_full_bypass_enabled", lambda: True)
    monkeypatch.setattr(model_mod, "memfabric_bypass_fence_enabled", lambda: True)
    monkeypatch.setattr(model_mod.torch.npu, "current_stream", lambda: stream)

    model_mod.DeepseekV41Model.engram_memfabric_diagnostic_fence(model, "runner-input-entry")
    model_mod.DeepseekV41Model.engram_memfabric_diagnostic_fence(model, "runner-after-pcp")

    assert model._engram_mf_runner_fence_count == 1
    assert stream.synchronize.call_count == 2

    monkeypatch.setattr(model_mod, "memfabric_full_bypass_enabled", lambda: False)
    model_mod.DeepseekV41Model.engram_memfabric_diagnostic_fence(model, "runner-input-entry")
    assert model._engram_mf_runner_fence_count == 1
    assert stream.synchronize.call_count == 2


def test_memfabric_staging_fp32_view_preserves_packed_row_layout():
    layout = MemFabricEngramLayout.build([(0, 5)], world_size=1)
    pool = SimpleNamespace(
        topology=SimpleNamespace(rank_id=0),
        layout=layout,
        entries=torch.zeros((layout.rows_per_slot, ENTRY_BYTES), dtype=torch.uint8),
    )
    view = MemFabricEngramTableView(pool, layout.segment(0), max_lookup_rows=3, device=torch.device("cpu"))

    assert view.staging_fp32.shape == (3, ENTRY_BYTES // 4)
    assert view.staging_fp32.data_ptr() == view.staging.data_ptr()

    scales = torch.arange(3 * SCALE_COUNT, dtype=torch.float32).view(3, SCALE_COUNT).add_(0.5)
    view.staging_fp32[:, CODE_BYTES // 4 :].copy_(scales)

    torch.testing.assert_close(view.staging[:, CODE_BYTES:].view(torch.float32), scales)


def test_memfabric_layout_folds_complete_tp_shards_into_one_uniform_grid():
    rank_rows = [[(0, 5), (1, 8)], [(0, 7), (1, 6)]]
    layout = MemFabricEngramLayout.build(
        rank_rows[0],
        world_size=2,
        rank_id=0,
        rank_layer_rows=rank_rows,
    )

    first = layout.segment(0)
    second = layout.segment(1)
    assert (first.rows, first.capacity, first.layer_base) == (5, 7, 0)
    assert (second.rows, second.capacity, second.layer_base) == (8, 8, 7)
    assert layout.rows_per_slot == 15
    assert layout.local_rows_span == 15

    # Rank 1's logical row 4 stays in rank 1's slot; TP rows are not striped.
    global_id = layout.rows_per_slot + first.layer_base + 4
    assert global_id == 19
    assert divmod(global_id, layout.rows_per_slot) == (1, 4)

    peer_layout = MemFabricEngramLayout.build(
        rank_rows[1],
        world_size=2,
        rank_id=1,
        rank_layer_rows=rank_rows,
    )
    assert peer_layout.rows_per_slot == layout.rows_per_slot
    assert peer_layout.local_rows_span == 13
    assert peer_layout.segment(0).capacity == first.capacity
    assert peer_layout.segment(1).capacity == second.capacity


def test_memfabric_replicated_mode_builds_independent_world1_topology(monkeypatch):
    monkeypatch.setattr(memfabric_mod, "_MEMFABRIC_REPLICATED", True)
    monkeypatch.setattr(memfabric_mod, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(memfabric_mod.torch.npu, "current_device", lambda: 11)

    topology = MemFabricEngramTopology.from_runtime(
        SimpleNamespace(
            store_port_base=8600,
            store_url_template="tcp://127.0.0.1:{port}",
        )
    )
    layout = MemFabricEngramLayout.build([(0, 5), (1, 8)], topology.world_size)

    assert (topology.rank_id, topology.world_size, topology.local_world_size) == (0, 1, 1)
    assert topology.replicated
    assert topology.pool_index == 3
    assert topology.store_url == "tcp://127.0.0.1:8611"
    assert layout.segment(0).capacity == 5
    assert layout.segment(1).capacity == 8
    assert layout.rows_per_slot == 13


def test_memfabric_tp_pool_builds_one_production_pool_per_dp_replica(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=3, world_size=8, cpu_group=object())
    dp_group = SimpleNamespace(rank_in_group=1, world_size=2)

    def gather_pool_metadata(values, value, *, group):
        assert group is tp_group.cpu_group
        values[:] = [value] * tp_group.world_size

    monkeypatch.setattr(memfabric_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(memfabric_mod, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(memfabric_mod, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(memfabric_mod.dist, "all_gather_object", gather_pool_metadata)
    monkeypatch.setattr(memfabric_mod.torch.npu, "current_device", lambda: 11)

    topology = MemFabricEngramTopology.from_runtime(
        SimpleNamespace(
            pool_scope="tp_group",
            store_port_base=8600,
            store_url_template="tcp://127.0.0.1:{port}/{pool_index}/{tp_rank}",
        )
    )

    assert (topology.rank_id, topology.world_size, topology.local_world_size) == (3, 8, 8)
    assert topology.pool_scope == "tp_group"
    assert topology.pool_index == 1
    assert topology.tp_rank == 3
    assert topology.store_url == "tcp://127.0.0.1:8601/1/3"


def test_memfabric_topology_rejects_dp_axis_pool_scope():
    with pytest.raises(ValueError, match="only supports.*tp_group"):
        MemFabricEngramTopology.from_runtime(
            SimpleNamespace(
                pool_scope="engram_share_group",
                store_port_base=8600,
                store_url_template="tcp://127.0.0.1:{port}",
            )
        )


def test_memfabric_no_gva_mode_keeps_model_side_pool_without_native_init(monkeypatch):
    monkeypatch.setattr(memfabric_mod, "_MEMFABRIC_GVA_MODE", "none")
    monkeypatch.setattr(
        memfabric_mod.torch.npu,
        "current_stream",
        lambda *_: SimpleNamespace(npu_stream=123),
    )
    monkeypatch.setattr(
        memfabric_mod,
        "_load_offload_module",
        MagicMock(side_effect=AssertionError("no-GVA mode must not import acc_offload")),
    )
    topology = MemFabricEngramTopology(0, 2, 2, 0, 0, "tcp://127.0.0.1:8600")
    rank_rows = [[(0, 5)], [(0, 7)]]
    layout = MemFabricEngramLayout.build(
        rank_rows[0], world_size=2, rank_layer_rows=rank_rows
    )

    pool = MemFabricEngramPool(topology, layout)

    assert not pool.initialized
    assert pool.offload is None
    assert pool.pointer == 0
    assert pool.mapped_reserve_size == 0
    assert pool.entries.shape == (0, ENTRY_BYTES)
    pool.close()


def test_memfabric_minimal_gva_mode_maps_one_gib_without_table(monkeypatch):
    calls = []
    offload = SimpleNamespace(
        Scene=SimpleNamespace(LOCAL=1, SHARED=2),
        OffloadConfig=SimpleNamespace,
        initialize=lambda config: calls.append(("initialize", config)) or 0,
        malloc=MagicMock(side_effect=AssertionError("minimal GVA must not call malloc")),
        register_entry_table=MagicMock(
            side_effect=AssertionError("minimal GVA must not register an entry table")
        ),
        uninitialize=lambda: calls.append(("uninitialize", None)) or 0,
    )
    monkeypatch.setattr(memfabric_mod, "_MEMFABRIC_GVA_MODE", "minimal")
    monkeypatch.setattr(
        memfabric_mod.torch.npu,
        "current_stream",
        lambda *_: SimpleNamespace(npu_stream=123),
    )
    monkeypatch.setattr(memfabric_mod, "_load_offload_module", lambda: offload)
    monkeypatch.setattr(memfabric_mod, "_load_acl_context_api", object)
    monkeypatch.setattr(memfabric_mod, "_get_acl_current_context", lambda _: 1)
    monkeypatch.setattr(memfabric_mod, "_restore_acl_context", lambda *_: (1, False))
    monkeypatch.setattr(memfabric_mod, "_probe_tp_after_memfabric_init_stage", lambda *_: None)
    topology = MemFabricEngramTopology(1, 2, 2, 8, 0, "tcp://127.0.0.1:8600")
    rank_rows = [[(0, 5)], [(0, 7)]]
    layout = MemFabricEngramLayout.build(
        rank_rows[1],
        world_size=2,
        rank_id=1,
        rank_layer_rows=rank_rows,
    )

    pool = MemFabricEngramPool(topology, layout)

    config = calls[0][1]
    assert calls[0][0] == "initialize"
    assert config.world_size == 2
    assert config.rank_id == 1
    assert config.reserve_size == memfabric_mod.ONE_GIB
    assert config.alloc_size == memfabric_mod.ONE_GIB
    assert pool.initialized
    assert pool.pointer == 0
    assert pool.entries.shape == (0, ENTRY_BYTES)
    pool.close()
    assert calls[-1][0] == "uninitialize"


def test_memfabric_layout_rejects_uint32_entry_id_overflow():
    with pytest.raises(ValueError, match="exceeds uint32"):
        rows = [(0, 1 << 32)]
        MemFabricEngramLayout.build(rows, world_size=2, rank_layer_rows=[rows, rows])


def test_table_view_packs_complete_local_tp_shard():
    view = object.__new__(MemFabricEngramTableView)
    view.segment = SimpleNamespace(rows=7)
    view.local_entries = torch.zeros((7, ENTRY_BYTES), dtype=torch.uint8)

    codes = torch.arange(5 * CODE_BYTES, dtype=torch.int64).remainder(251).sub(125).to(torch.int8)
    codes = codes.view(5, CODE_BYTES)
    scales = torch.arange(5 * SCALE_COUNT, dtype=torch.float32).view(5, SCALE_COUNT).add_(0.5)
    view.copy_entries(local_row_start=1, codes=codes, scales=scales)

    assert torch.equal(view.local_entries[1:6, :CODE_BYTES], codes.view(torch.uint8))
    assert torch.equal(
        view.local_entries[1:6, CODE_BYTES:],
        scales.contiguous().view(torch.uint8).reshape(5, SCALE_BYTES),
    )
    assert torch.count_nonzero(view.local_entries[[0, 6]]) == 0


def test_zero_sized_loader_sentinels_populate_memfabric_packed_entries(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    codes = torch.arange(11 * 256, dtype=torch.int64).remainder(251).sub(125).to(torch.int8).view(11, 256)
    scales = torch.arange(11 * 8, dtype=torch.float32).view(11, 8).add_(1)
    save_file({key: codes, scale_key: scales}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )

    stored_codes = torch.empty_like(codes)
    stored_scales = torch.empty_like(scales)
    view = SimpleNamespace(copy_entries=MagicMock(), pool=SimpleNamespace(load_barrier=MagicMock()))

    def copy_entries(start, code_chunk, scale_chunk):
        stop = start + code_chunk.shape[0]
        stored_codes[start:stop].copy_(code_chunk)
        stored_scales[start:stop].copy_(scale_chunk)

    view.copy_entries.side_effect = copy_entries
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
    table.weight_scale_inv = torch.nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
    table.bind_checkpoint(tmp_path, key)

    table._weight_loader(table.weight, torch.empty(0))

    torch.testing.assert_close(stored_codes, codes)
    torch.testing.assert_close(stored_scales, scales)
    assert table.weight_scale_inv.numel() == 0
    view.pool.load_barrier.assert_called_once_with(None)


def test_memfabric_skip_load_keeps_collective_order_without_touching_storage(monkeypatch):
    view = SimpleNamespace(pool=SimpleNamespace(load_barrier=MagicMock()))
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    table._memfabric_table = view
    table._load_into_storage = MagicMock()
    monkeypatch.setattr(embedding_mod, "_ENGRAM_MF_SKIP_LOAD", True)

    table.load_checkpoint("unused-model-path", "layers.1.engram.embed.weight")

    table._load_into_storage.assert_not_called()
    view.pool.load_barrier.assert_called_once_with(None)


def test_memfabric_load_barrier_uses_tp_pool_group(monkeypatch):
    cpu_group = object()
    tp_group = SimpleNamespace(world_size=8, cpu_group=cpu_group)
    pool = object.__new__(MemFabricEngramPool)
    pool.topology = SimpleNamespace(world_size=8)

    def gather_errors(errors, error, *, group):
        assert group is cpu_group
        errors[:] = [error] * tp_group.world_size

    monkeypatch.setattr(memfabric_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(memfabric_mod.dist, "all_gather_object", gather_errors)

    pool.load_barrier(None)


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
    assert memfabric.call_args.args[1] is ids
    host_uva.assert_not_called()


def test_entry_gather_uses_memfabric_tensor_api():
    device = object()
    staging = SimpleNamespace(device=device)
    global_ids = object()
    count = object()
    offload = SimpleNamespace(entry_gather=MagicMock(return_value=0))
    table = SimpleNamespace(
        staging=staging,
        global_ids=global_ids,
        count=count,
        pool=SimpleNamespace(
            offload=offload,
            topology=SimpleNamespace(device_id=6),
        ),
    )

    assert npu_mod._submit_memfabric_entry_gather(table) == 0

    offload.entry_gather.assert_called_once_with(staging, global_ids, count, device)


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
    pool.entries = torch.empty(0)
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
