# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Order HCCL and AICPU initialization before MemFabric Engram mapping."""

from __future__ import annotations

import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.utils import enable_custom_op

_WARMUP_NUM_HEADS_Q = 64
_WARMUP_NUM_HEADS_KV = 1
_WARMUP_HEAD_DIM = 512
_WARMUP_BATCH_SIZE = 4
_WARMUP_MAX_SEQLEN_Q = 1024
_WARMUP_MAX_SEQLEN_ORI_KV = 1024
_WARMUP_MAX_SEQLEN_CMP_KV = 1024
_WARMUP_CMP_RATIO = 128
_WARMUP_ORI_MASK_MODE = 4
_WARMUP_CMP_MASK_MODE = 3
_WARMUP_ORI_WIN_LEFT = 127
_WARMUP_ORI_WIN_RIGHT = 0


def _uses_memfabric_engram(vllm_config: object) -> bool:
    """Return whether model construction will create a MemFabric Engram pool."""
    engram_config = getattr(vllm_config, "engram_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None:
        text_config = getattr(model_config, "hf_config", None)
    return bool(
        get_ascend_config().engram_offload_backend == "memfabric"
        and getattr(engram_config, "cpu_offload", False)
        and getattr(text_config, "engram_layer_ids", ())
    )


def _materialize_hccl_group(label: str, coordinator: object, device: torch.device) -> str:
    """Create the lazy HCCL communicator and return its name for logging."""
    group = coordinator.device_group
    local_rank = torch.distributed.get_rank(group=group)
    backend = group._get_backend(device)
    name = backend.get_hccl_comm_name(local_rank)
    logger.info("Engram MemFabric ordered warmup: %s HCCL communicator ready: %s", label, name)
    return name


def _warm_up_sparse_flash_mla_metadata(device: torch.device) -> None:
    """Launch one supported AICPU metadata operation and wait for completion."""
    # This is the supported SparseFlashMlaMetadata AICPU example shape. The
    # result is intentionally unused: completing one launch is what registers
    # the AICPU operator before DeepSeekV41Model creates the MemFabric pool.
    torch.npu.synchronize(device)
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        _WARMUP_NUM_HEADS_Q,
        _WARMUP_NUM_HEADS_KV,
        _WARMUP_HEAD_DIM,
        batch_size=_WARMUP_BATCH_SIZE,
        max_seqlen_q=_WARMUP_MAX_SEQLEN_Q,
        max_seqlen_ori_kv=_WARMUP_MAX_SEQLEN_ORI_KV,
        max_seqlen_cmp_kv=_WARMUP_MAX_SEQLEN_CMP_KV,
        cmp_ratio=_WARMUP_CMP_RATIO,
        ori_mask_mode=_WARMUP_ORI_MASK_MODE,
        cmp_mask_mode=_WARMUP_CMP_MASK_MODE,
        ori_win_left=_WARMUP_ORI_WIN_LEFT,
        ori_win_right=_WARMUP_ORI_WIN_RIGHT,
        layout_q="BSND",
        layout_kv="BSND",
    )
    torch.npu.synchronize(device)
    del metadata


def warm_up_before_memfabric_engram(vllm_config: object, device: torch.device) -> bool:
    """Warm AICPU before the model creates its full MemFabric peer mapping.

    SparseFlashMlaMetadata is an AICPU custom operator. On A3, registering it
    after a full MemFabric peer mapping can fail in ``halMemBindSibling``.
    Every worker must materialize MC2 and EP in this order before this launch.
    """
    if not _uses_memfabric_engram(vllm_config):
        return False
    if not enable_custom_op():
        raise RuntimeError("MemFabric Engram ordered warmup requires Ascend custom operators")

    backend_device = torch.device("npu")
    _materialize_hccl_group("mc2", get_mc2_group(), backend_device)
    _materialize_hccl_group("ep", get_ep_group(), backend_device)
    _warm_up_sparse_flash_mla_metadata(device)
    logger.info("Engram MemFabric ordered AICPU warmup complete")
    return True
