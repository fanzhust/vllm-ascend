# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig, EngramMemFabricConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_engram_overlap_hardware_default_and_explicit_override(monkeypatch, device, explicit):
    monkeypatch.setattr("vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(device))
    kwargs = {} if explicit is None else {"multistream_engram_overlap": explicit}
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)
    expected = device == AscendDeviceType.A5 if explicit is None else explicit
    assert config.multistream_engram_overlap is expected


def test_memfabric_config_uses_explicit_endpoints():
    config = EngramMemFabricConfig(store_url="tcp://10.0.0.1:8600", control_port=8599)
    assert config.num_nodes == 1
    assert config.node_rank == 0
    assert config.control_port == 8599


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"store_url": "tcp://host"}, "tcp://HOST:PORT"),
        ({"store_url": "tcp://host:8600/path"}, "tcp://HOST:PORT"),
        ({"control_port": 8600}, "different"),
        ({"control_port": 65536}, "valid TCP port"),
        ({"node_rank": 1}, "node_rank"),
        ({"num_nodes": 0}, "node_rank"),
    ],
)
def test_memfabric_config_rejects_invalid_store_settings(overrides, match):
    kwargs = {"store_url": "tcp://host:8600", "control_port": 8599} | overrides
    with pytest.raises(ValueError, match=match):
        EngramMemFabricConfig(**kwargs)
