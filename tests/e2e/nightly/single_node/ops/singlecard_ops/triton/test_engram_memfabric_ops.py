# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, model-free NPU tests for the MemFabric Engram lookup pipeline.

Run one boundary at a time so a device hang identifies the failing stage::

    pytest -q -s test_engram_memfabric_ops.py -k dequant_only
    pytest -q -s test_engram_memfabric_ops.py -k map_ids_only
    pytest -q -s test_engram_memfabric_ops.py -k entry_gather_only
    pytest -q -s test_engram_memfabric_ops.py -k full_pipeline_each_layer
    pytest -q -s test_engram_memfabric_ops.py -k full_pipeline_back_to_back

The first two tests do not initialize MemFabric. The remaining tests create a
single-rank SHARED pool whose minimum reservation is 1 GiB; override the NPU
and TCP store port with ``ENGRAM_MF_TEST_DEVICE`` and
``ENGRAM_MF_TEST_PORT`` when necessary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401 - registers torch.npu

from vllm_ascend.models.deepseek_v41.engram.memfabric import (
    CODE_BYTES,
    ENTRY_BYTES,
    SCALE_COUNT,
    MemFabricEngramLayout,
    MemFabricEngramPool,
    MemFabricEngramTableView,
    MemFabricEngramTopology,
)
from vllm_ascend.models.deepseek_v41.engram.npu import (
    MEMFABRIC_MAP_BLOCK,
    SCALE_GROUP,
    _submit_memfabric_entry_gather,
    gather_dequantize_memfabric,
)
from vllm_ascend.ops.triton.engram_lookup import (
    _engram_memfabric_dequant_kernel,
    _engram_memfabric_map_ids_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

TABLE_ROWS = 16
MAX_LOOKUP_ROWS = 64
VOCAB_START = 100


def _device() -> torch.device:
    device_id = int(os.getenv("ENGRAM_MF_TEST_DEVICE", "0"))
    torch.npu.set_device(device_id)
    return torch.device("npu", device_id)


def _codes_and_scales(rows: int, layer: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(rows * CODE_BYTES, dtype=torch.int64).view(rows, CODE_BYTES)
    codes = values.add(layer * 37).remainder(255).sub(127).to(torch.int8)
    powers = torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 0.0625, 0.03125], dtype=torch.float32)
    scales = powers.roll(layer).repeat(rows, 1)
    return codes, scales


def _pack_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows = codes.shape[0]
    packed = torch.empty((rows, ENTRY_BYTES), dtype=torch.uint8)
    packed[:, :CODE_BYTES].copy_(codes.view(torch.uint8))
    packed[:, CODE_BYTES:].copy_(scales.contiguous().view(torch.uint8).reshape(rows, SCALE_COUNT * 4))
    return packed


def _reference_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (codes.float().view(-1, SCALE_COUNT, SCALE_GROUP) * scales[:, :, None]).view(-1, CODE_BYTES).bfloat16()


def _sync(stage: str) -> None:
    print(f"[Engram MF op test] {stage} synchronize BEGIN", flush=True)
    torch.npu.synchronize()
    print(f"[Engram MF op test] {stage} synchronize DONE", flush=True)


@torch.inference_mode()
def test_dequant_only():
    """Run the Triton dequant kernel without MemFabric or entry_gather."""
    device = _device()
    init_device_properties_triton()
    tokens, local_heads, head_start, total_heads = 3, 2, 1, 4
    rows = tokens * local_heads
    ids_cpu = torch.tensor(
        [
            [0, 100, 101, 0],
            [0, 99, 105, 0],
            [0, 109, 110, 0],
        ],
        dtype=torch.int32,
    )
    assert ids_cpu.shape == (tokens, total_heads)
    codes, scales = _codes_and_scales(rows)
    staging = _pack_rows(codes, scales).view(torch.int8).to(device)
    staging_fp32 = staging.view(torch.float32)
    ids = ids_cpu.to(device)
    output = torch.empty((tokens, local_heads, CODE_BYTES), dtype=torch.bfloat16, device=device)

    print("[Engram MF op test] dequant launch", flush=True)
    _engram_memfabric_dequant_kernel[(rows,)](
        staging,
        staging_fp32,
        ids,
        output,
        rows,
        VOCAB_START,
        VOCAB_START + 10,
        ids.stride(0),
        ENTRY_BYTES=ENTRY_BYTES,
        CODE_BYTES=CODE_BYTES,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=local_heads,
        num_warps=4,
    )
    _sync("dequant-only")

    expected = _reference_rows(codes, scales)
    selected = ids_cpu[:, head_start : head_start + local_heads].reshape(-1)
    owned = (selected >= VOCAB_START) & (selected < VOCAB_START + 10)
    expected[~owned] = 0
    torch.testing.assert_close(output.cpu().view(rows, CODE_BYTES), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_map_ids_only():
    """Run the vectorized Triton id mapper without MemFabric."""
    device = _device()
    init_device_properties_triton()
    tokens, local_heads, head_start = 3, 2, 1
    rows = tokens * local_heads
    slot_rank, layer_base, rows_per_slot = 3, 7, 13
    ids_cpu = torch.tensor(
        [
            [0, 100, 101, 0],
            [0, 99, 105, 0],
            [0, 109, 110, 0],
        ],
        dtype=torch.int32,
    )
    ids = ids_cpu.to(device)
    global_ids = torch.full((rows,), -1, dtype=torch.int64, device=device)
    count = torch.full((1,), -1, dtype=torch.int32, device=device)
    grid = ((rows + MEMFABRIC_MAP_BLOCK - 1) // MEMFABRIC_MAP_BLOCK,)

    print("[Engram MF op test] map_ids launch", flush=True)
    _engram_memfabric_map_ids_kernel[grid](
        ids,
        global_ids,
        count,
        rows,
        VOCAB_START,
        VOCAB_START + 10,
        ids.stride(0),
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        SLOT_RANK=slot_rank,
        LAYER_BASE=layer_base,
        ROWS_PER_SLOT=rows_per_slot,
        BLOCK=MEMFABRIC_MAP_BLOCK,
        num_warps=1,
    )
    _sync("map-ids-only")

    selected = ids_cpu[:, head_start : head_start + local_heads].reshape(-1).to(torch.int64)
    owned = (selected >= VOCAB_START) & (selected < VOCAB_START + 10)
    local = torch.where(owned, selected - VOCAB_START, 0)
    expected = slot_rank * rows_per_slot + layer_base + local
    torch.testing.assert_close(global_ids.cpu(), expected, rtol=0, atol=0)
    assert count.cpu().item() == rows


@dataclass
class _PoolFixture:
    pool: MemFabricEngramPool
    views: tuple[MemFabricEngramTableView, MemFabricEngramTableView]
    codes: tuple[torch.Tensor, torch.Tensor]
    scales: tuple[torch.Tensor, torch.Tensor]
    device: torch.device


@pytest.fixture(scope="module")
def mf_pool() -> _PoolFixture:
    pytest.importorskip("memfabric_hybrid")
    device = _device()
    device_id = device.index
    port = int(os.getenv("ENGRAM_MF_TEST_PORT", "28606"))
    layout = MemFabricEngramLayout.build([(0, TABLE_ROWS), (1, TABLE_ROWS)], world_size=1)
    topology = MemFabricEngramTopology(
        rank_id=0,
        world_size=1,
        local_world_size=1,
        device_id=device_id,
        pool_index=0,
        store_url=f"tcp://127.0.0.1:{port}",
    )
    print(f"[Engram MF op test] pool initialize device={device_id} port={port}", flush=True)
    pool = MemFabricEngramPool(topology, layout)
    views = tuple(
        MemFabricEngramTableView(pool, layout.segment(layer), MAX_LOOKUP_ROWS, device) for layer in (0, 1)
    )
    codes_and_scales = tuple(_codes_and_scales(TABLE_ROWS, layer) for layer in (0, 1))
    for layer, view in enumerate(views):
        codes, scales = codes_and_scales[layer]
        view.copy_entries(0, codes, scales)
    fixture = _PoolFixture(
        pool=pool,
        views=views,
        codes=tuple(pair[0] for pair in codes_and_scales),
        scales=tuple(pair[1] for pair in codes_and_scales),
        device=device,
    )
    try:
        yield fixture
    finally:
        _sync("pool-close")
        for view in views:
            view.close()
        pool.close()
        print("[Engram MF op test] pool closed", flush=True)


@torch.inference_mode()
def test_entry_gather_only(mf_pool: _PoolFixture):
    """Gather known packed rows without either Triton kernel."""
    view = mf_pool.views[0]
    global_ids_cpu = torch.tensor([0, 3, 15, 16, 20, 31], dtype=torch.int64)
    rows = global_ids_cpu.numel()
    view.global_ids[:rows].copy_(global_ids_cpu.to(mf_pool.device))
    view.count.fill_(rows)
    _sync("entry-gather-inputs")

    print("[Engram MF op test] entry_gather call BEGIN", flush=True)
    rc = _submit_memfabric_entry_gather(view)
    print(f"[Engram MF op test] entry_gather call RETURN rc={rc}", flush=True)
    assert rc == 0
    _sync("entry-gather-only")

    expected = mf_pool.pool.entries.index_select(0, global_ids_cpu).clone()
    actual = view.staging[:rows].view(torch.uint8).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _pipeline_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids_cpu = torch.tensor(
        [
            [100, 101, 99],
            [105, 110, 115],
            [131, 116, 102],
            [103, 114, 200],
        ],
        dtype=torch.int32,
    )
    return ids_cpu, ids_cpu.to(device)


def _pipeline_expected(fixture: _PoolFixture, layer: int, ids_cpu: torch.Tensor) -> torch.Tensor:
    selected = ids_cpu[:, :2].reshape(-1).to(torch.int64)
    owned = (selected >= VOCAB_START) & (selected < VOCAB_START + TABLE_ROWS)
    local = torch.where(owned, selected - VOCAB_START, 0)
    decoded = _reference_rows(fixture.codes[layer], fixture.scales[layer])
    expected = decoded.index_select(0, local).clone()
    expected[~owned] = 0
    return expected.view(ids_cpu.shape[0], 2, CODE_BYTES)


def _submit_full_pipeline(
    fixture: _PoolFixture, layer: int, ids: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((ids.shape[0], 2, CODE_BYTES), dtype=torch.bfloat16, device=fixture.device)
    print(f"[Engram MF op test] full pipeline layer={layer} BEGIN", flush=True)
    gather_dequantize_memfabric(
        fixture.views[layer],
        ids,
        head_start=0,
        local_heads=2,
        pad_heads=2,
        output=output,
        vocab_start=VOCAB_START,
        vocab_end=VOCAB_START + TABLE_ROWS,
    )
    print(f"[Engram MF op test] full pipeline layer={layer} SUBMITTED", flush=True)
    return output


@torch.inference_mode()
def test_full_pipeline_each_layer(mf_pool: _PoolFixture):
    """Synchronize and validate each layer independently."""
    ids_cpu, ids = _pipeline_inputs(mf_pool.device)
    for layer in (0, 1):
        output = _submit_full_pipeline(mf_pool, layer, ids)
        _sync(f"full-pipeline-layer-{layer}")
        torch.testing.assert_close(output.cpu(), _pipeline_expected(mf_pool, layer, ids_cpu), rtol=0, atol=0)


@torch.inference_mode()
def test_full_pipeline_back_to_back(mf_pool: _PoolFixture):
    """Submit both Engram layers before synchronizing, like model forward."""
    ids_cpu, ids = _pipeline_inputs(mf_pool.device)
    outputs = [_submit_full_pipeline(mf_pool, layer, ids) for layer in (0, 1)]
    _sync("full-pipeline-back-to-back")
    for layer, output in enumerate(outputs):
        torch.testing.assert_close(output.cpu(), _pipeline_expected(mf_pool, layer, ids_cpu), rtol=0, atol=0)
