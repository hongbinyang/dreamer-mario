"""Tiny YAML config loader with attribute access and CLI overrides."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import yaml


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def ns_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(ns).items()}
    return ns


def dict_to_ns(d):
    return _to_ns(d)


def _set_dotted(d: dict, key: str, value: str):
    parts = key.split(".")
    node = d
    for p in parts[:-1]:
        node = node[p]
    old = node[parts[-1]]
    # Cast the override to the type of the existing value.
    if isinstance(old, bool):
        node[parts[-1]] = value.lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        node[parts[-1]] = int(value)
    elif isinstance(old, float):
        node[parts[-1]] = float(value)
    else:
        node[parts[-1]] = value


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply a list of 'dotted.key=value' strings to a raw config dict, in place."""
    for override in overrides:
        key, value = override.split("=", 1)
        _set_dotted(raw, key, value)
    return raw


def load_yaml(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return apply_overrides(raw, overrides or [])


def load_config(argv: list[str] | None = None):
    """Usage: python evaluate.py --config configs/default.yaml \
                 --set train.total_steps=200000 --set env.grayscale=true"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                        help="dotted.key=value overrides")
    args = parser.parse_args(argv)
    return _to_ns(load_yaml(args.config, args.set))


def _xla_device(required: bool):
    """TPU support via torch_xla (https://github.com/pytorch/xla) -- an
    optional, separately-installed package, never a hard dependency of this
    project. Untested on real TPU hardware: this repo was developed
    entirely on Apple Silicon/MPS. If training doesn't actually speed up on
    a TPU, the likely culprit is dreamer/rssm.py's `imagine()`, a plain
    Python loop; XLA's lazy tracing may need explicit xm.mark_step() calls
    per iteration to compile well."""
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        if required:
            raise ImportError(
                "--device tpu requires torch_xla, which isn't installed. "
                "See https://github.com/pytorch/xla for setup instructions."
            ) from None
        return None
    return xm.xla_device()


def pick_device(name: str):
    """name: 'auto' | 'cpu' | 'cuda'[:N] | 'mps' | 'tpu'.

    'auto' probes accelerators in order (cuda, mps, tpu) and falls back to
    cpu. Anything else is passed straight to torch.device(), so e.g.
    'cuda:1' for a specific GPU still works.
    """
    import os
    import torch

    if name == "tpu":
        device = _xla_device(required=True)
    elif name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = _xla_device(required=False) or torch.device("cpu")
    else:
        device = torch.device(name)

    if device.type == "mps":
        # Not every op is implemented on MPS yet; fall back to CPU for just
        # those instead of crashing. setdefault so an explicit "0" set
        # beforehand (to get hard failures while debugging) isn't clobbered.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return device
