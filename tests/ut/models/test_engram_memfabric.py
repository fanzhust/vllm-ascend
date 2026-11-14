# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Python contracts for model-owned MemFabric Engram storage."""

import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend import envs
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
        def __init__(self, layer_rows, max_tokens, memfabric_config, parallel, checkpoint_identity):
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
        model_mod.DeepseekV41Model(vllm_config=_early_init_config(dp_shared_memory=False))

    assert [event[0] for event in events] == ["session", "tensor-allocation"]
    assert events[0][1] == [(0, (10,) * 8)]
    assert events[0][2] == 8192
    assert events[0][3] is ascend_config.engram_memfabric


def test_memfabric_entry_contains_codes_and_fp32_scales():
    assert CODE_BYTES == 256
    assert SCALE_COUNT == 8
    assert SCALE_BYTES == 32
    assert ENTRY_BYTES == 288


def _mock_worker_world(monkeypatch, node_ranks, worker_rank, tp):
    world = SimpleNamespace(world_size=len(node_ranks), rank_in_group=worker_rank, cpu_group=object())

    def gather(output, node_rank, *, group):
        assert group is world.cpu_group
        assert node_rank == node_ranks[worker_rank]
        output[:] = node_ranks

    monkeypatch.setattr(memfabric_mod, "get_world_group", lambda: world)
    monkeypatch.setattr(memfabric_mod, "get_tensor_model_parallel_rank", lambda: worker_rank % tp)
    monkeypatch.setattr(memfabric_mod.dist, "all_gather_object", gather)


@pytest.mark.parametrize(
    "node_ranks,worker_rank,local_size,local_rank",
    [
        ([0] * 16, 13, 16, 13),  # One engine launches all DP replicas on one machine.
        ([0] * 8 + [1] * 8, 13, 8, 5),  # Offline DP spans two machines.
        ([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4, 13, 8, 5),  # Both TP groups span two machines.
    ],
)
def test_memfabric_control_uses_worker_placement(monkeypatch, node_ranks, worker_rank, local_size, local_rank):
    _mock_worker_world(monkeypatch, node_ranks, worker_rank, tp=8)
    store = MagicMock()
    monkeypatch.setattr(memfabric_mod.dist, "TCPStore", store)
    node_rank = node_ranks[worker_rank]
    control = memfabric_mod.EngramPoolControl(
        SimpleNamespace(
            store_url="tcp://127.0.0.1:8600", control_port=8599, num_nodes=len(set(node_ranks)), node_rank=node_rank
        ),
        SimpleNamespace(tensor_parallel_size=8, data_parallel_size=2, nnodes=1),
    )

    assert (control.local_size, control.local_rank, control.world_size) == (local_size, local_rank, 16)
    assert control.rank == node_rank * local_size + local_rank
    assert store.call_args.kwargs["is_master"] is False


def test_memfabric_control_rendezvous_waits_for_late_worker_registration(monkeypatch):
    timeout = timedelta(seconds=10)
    monkeypatch.setattr(memfabric_mod, "_STARTUP_TIMEOUT", timeout)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        control_port = listener.getsockname()[1]
    controls = []
    for node, tp, dp in ((0, 4, 4), (1, 2, 8)):
        for dp_rank in range(dp):
            for tp_rank in range(tp):
                _mock_worker_world(monkeypatch, [node] * (dp * tp), dp_rank * tp + tp_rank, tp)
                controls.append(
                    memfabric_mod.EngramPoolControl(
                        SimpleNamespace(
                            store_url="tcp://127.0.0.1:8600", control_port=control_port, num_nodes=2, node_rank=node
                        ),
                        SimpleNamespace(
                            tensor_parallel_size=tp,
                            local_world_size=tp,
                            data_parallel_size_local=1,
                            data_parallel_size=dp,
                            data_parallel_rank_local=0,
                            data_parallel_index=dp_rank,
                            nnodes=1,
                            data_parallel_external_lb=True,
                            data_parallel_hybrid_lb=False,
                        ),
                    )
                )
    with ThreadPoolExecutor(max_workers=len(controls)) as executor:
        first = executor.submit(controls[0].plan, "checkpoint", None)
        controls[1].store.wait(["worker/0/0"], timeout)
        others = [executor.submit(control.plan, "checkpoint", None) for control in controls[1:]]
        for worker, result in zip(controls, [first, *others], strict=True):
            num_pools, membership = result.result(timeout=timeout.total_seconds())
            assert num_pools == 2
            assert len(membership.members) == 16
            assert membership.local_world_size == 8
            assert membership.members[membership.rank_id] == worker.rank
            assert membership.rank_id // 8 == worker.node_rank


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
        lambda: SimpleNamespace(engram_memfabric=object()),
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


def test_memfabric_staging_fp32_view_preserves_packed_row_layout():
    layout = MemFabricEngramLayout.build([(0, (5,))], 1, 0, 1, 0)
    pool = SimpleNamespace(
        membership=SimpleNamespace(members=(0,), rank_id=0),
        layout=layout,
        use_fused_lookup=False,
        entries=torch.zeros((layout.rows_per_slot, ENTRY_BYTES), dtype=torch.uint8),
    )
    view = MemFabricEngramTableView(pool, layout.segments[0], max_lookup_rows=3, device=torch.device("cpu"))

    assert view.staging_fp32.shape == (3, ENTRY_BYTES // 4)
    assert view.staging_fp32.data_ptr() == view.staging.data_ptr()

    scales = torch.arange(3 * SCALE_COUNT, dtype=torch.float32).view(3, SCALE_COUNT).add_(0.5)
    view.staging_fp32[:, CODE_BYTES // 4 :].copy_(scales)

    torch.testing.assert_close(view.staging[:, CODE_BYTES:].view(torch.float32), scales)


@pytest.mark.parametrize(("value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_memfabric_fused_lookup_env(monkeypatch, value, expected):
    name = "VLLM_ASCEND_ENGRAM_MF_FUSED"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert envs.VLLM_ASCEND_ENGRAM_MF_FUSED is expected


def test_memfabric_fused_lookup_rejects_old_wheel_before_mapping(monkeypatch):
    control = MagicMock(tp=1)
    control.plan.return_value = (1, object())
    control.exchange_errors.side_effect = RuntimeError("layout rejected")
    pool = MagicMock()
    monkeypatch.setenv("VLLM_ASCEND_ENGRAM_MF_FUSED", "1")
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=object()))
    monkeypatch.setattr(memfabric_mod, "EngramPoolControl", lambda *_: control)
    monkeypatch.setattr(memfabric_mod, "MemFabricEngramPool", pool)

    with pytest.raises(RuntimeError, match="layout rejected"):
        memfabric_mod.MemFabricEngramSession([(1, (10,))], 4, object(), object(), lambda: "checkpoint")

    stage, error = control.exchange_errors.call_args.args
    assert stage == "layout"
    assert "requires a wheel with engram_entry_gather support" in error
    pool.assert_not_called()


def test_memfabric_fused_lookup_skips_legacy_scratch_and_reuses_descriptor():
    layout = MemFabricEngramLayout.build([(1, (10, 11, 20, 23)), (14, (14, 17, 19, 24))], 2, 1, 2, 1)
    offload = SimpleNamespace(EngramLookupDesc=MagicMock(), engram_entry_gather=MagicMock(return_value=0))
    pool = SimpleNamespace(
        membership=SimpleNamespace(members=(0, 1), rank_id=1),
        layout=layout,
        entries=torch.zeros((layout.rows_per_slot, ENTRY_BYTES), dtype=torch.uint8),
        offload=offload,
        use_fused_lookup=True,
        ready=True,
    )
    view = MemFabricEngramTableView(pool, layout.segments[1], max_lookup_rows=3, device=torch.device("cpu"))
    view.configure_fused_lookup(head_start=3, local_heads=1, vocab_start=50, vocab_end=74)
    offload.EngramLookupDesc.assert_called_once_with(
        head_start=3,
        local_heads=1,
        vocab_start=50,
        vocab_end=74,
        pool_row_base=31,
        owner_rows=21,
        extra_owners=1,
        layer_base=22,
    )
    assert view.global_ids is view.count is view.staging is view.staging_fp32 is None

    # Same non-contiguous per-layer view produced by the actual hash path.
    ids = torch.empty((3, 2, 4), dtype=torch.int32)[:, 1]
    output = torch.empty((3, CODE_BYTES), dtype=torch.bfloat16)
    kwargs = dict(head_start=3, local_heads=1, pad_heads=1, output=output, vocab_start=50, vocab_end=74)
    assert npu_mod.gather_dequantize_memfabric(view, ids, **kwargs) is output
    forwarded_ids, desc, out = offload.engram_entry_gather.call_args.args
    assert forwarded_ids is ids
    assert desc is view.fused_lookup_desc
    assert out.shape == (3, 1, CODE_BYTES)
    assert out.data_ptr() == output.data_ptr()

    offload.engram_entry_gather.return_value = -1
    with pytest.raises(RuntimeError, match="Fused MemFabric Engram lookup failed: rc=-1"):
        npu_mod.gather_dequantize_memfabric(view, ids, **kwargs)
    view.close()
    assert view.fused_lookup_desc is None


def test_memfabric_allows_cross_node_tp_but_host_uva_does_not(monkeypatch):
    tp_group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(embedding_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: None)
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda _: [True, False])

    assert embedding_mod._validate_engram_node_placement(object()) is None
    with pytest.raises(ValueError, match="TP ranks.*single node"):
        embedding_mod._validate_engram_node_placement(None)


def test_table_view_packs_owned_rows():
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


def test_checkpoint_load_writes_rows_owned_outside_the_workers_heads(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    codes = torch.arange(11 * 256, dtype=torch.int64).remainder(251).sub(125).to(torch.int8).view(11, 256)
    scales = torch.arange(11 * 8, dtype=torch.float32).view(11, 8).add_(1)
    save_file({key: codes, scale_key: scales}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )

    stored_codes = torch.empty_like(codes[5:8])
    stored_scales = torch.empty_like(scales[5:8])
    view = SimpleNamespace(copy_entries=MagicMock(), segment=SimpleNamespace(owned_start=5, rows=3))

    def copy_entries(start, code_chunk, scale_chunk):
        assert code_chunk.device.type == scale_chunk.device.type == "cpu"
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
    table.vocab_end_idx = 5
    table.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
    table.bind_checkpoint(tmp_path, key)

    # Model construction supplies a non-CPU default device to this loader.
    with torch.device("meta"):
        table.load_checkpoint(tmp_path, key, chunk_rows=2)
    torch.testing.assert_close(stored_codes, codes[5:8])
    torch.testing.assert_close(stored_scales, scales[5:8])
    view.copy_entries.reset_mock()

    table._weight_loader(table.weight, torch.empty(0))
    view.copy_entries.assert_not_called()


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


def test_pool_close_frees_backing_before_uninitializing():
    calls = []

    pool = object.__new__(MemFabricEngramPool)
    pool.offload = SimpleNamespace(
        free=lambda pointer, flags: calls.append(("free", pointer, flags)),
        uninitialize=lambda: calls.append(("uninitialize",)),
    )
    pool.pointer = 1234
    pool.entries = torch.empty(0)
    pool._buffer = SimpleNamespace()
    pool.initialized = True

    pool.close()
    pool.close()
    assert pool.pointer == 0
    assert not pool.initialized
    assert calls == [("free", 1234, 0), ("uninitialize",)]
