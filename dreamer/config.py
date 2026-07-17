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


def load_config(argv: list[str] | None = None):
    """Usage: python train.py --config configs/default.yaml \
                 --set train.total_steps=200000 --set env.grayscale=true"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                        help="dotted.key=value overrides")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    for override in args.set:
        key, value = override.split("=", 1)
        _set_dotted(raw, key, value)
    cfg = _to_ns(raw)
    cfg.resume = args.resume
    return cfg


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
