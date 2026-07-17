import os

import pytest
import torch

from dreamer.config import pick_device


def test_pick_device_explicit_cpu():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_passes_through_unknown_names_to_torch():
    # e.g. "cuda:1" for a specific GPU -- torch.device() construction never
    # validates hardware availability, so this works even without a GPU.
    assert pick_device("cuda:1") == torch.device("cuda:1")


def test_pick_device_auto_falls_back_to_cpu_without_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto") == torch.device("cpu")


def test_pick_device_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("auto") == torch.device("cuda")


def test_pick_device_tpu_without_torch_xla_raises_clear_error():
    with pytest.raises(ImportError, match="torch_xla"):
        pick_device("tpu")


def test_pick_device_auto_tpu_probe_is_optional(monkeypatch):
    # No CUDA/MPS and no torch_xla installed -> should fall back to CPU
    # quietly, not raise, since TPU is optional under "auto".
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto") == torch.device("cpu")


def test_pick_device_mps_sets_fallback_env_var(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert pick_device("mps") == torch.device("mps")
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_pick_device_mps_does_not_clobber_explicit_fallback_setting(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    pick_device("mps")
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
