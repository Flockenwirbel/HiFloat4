#!/usr/bin/env python3
"""Verify Wan config.json and derive init kwargs from legacy config.

Usage: python scripts/verify_wan_config.py /path/to/wan_model_dir
"""
import json
import os
import sys
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_wan_config.py /path/to/wan_model_dir")
        sys.exit(2)

    load_dir = sys.argv[1]
    cfg_path = os.path.join(load_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"config.json not found at: {cfg_path}")
        sys.exit(1)

    raw_cfg = read_json(cfg_path)
    print("Loaded config.json keys:", ", ".join(sorted(raw_cfg.keys())))

    # detect legacy
    legacy_keys = {"in_dim", "out_dim", "num_heads", "dim"}
    is_legacy = any(k in raw_cfg for k in legacy_keys)
    print("Legacy-style config detected:" , is_legacy)

    # attempt to build init kwargs using existing loader if available
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, repo_root)
        from src.hifloat4.wan_model_utils import _load_wan_weight_index, _build_legacy_wan_init_kwargs

        weight_map = None
        try:
            weight_map = _load_wan_weight_index(load_dir)
        except Exception as e:
            print("Could not load safetensors index:", e)

        if weight_map is not None:
            init_kwargs = _build_legacy_wan_init_kwargs(load_dir, raw_cfg, weight_map)
            print("Derived init kwargs:")
            for k, v in sorted(init_kwargs.items()):
                print(f"  {k}: {v}")
        else:
            print("No safetensors index available; cannot derive full init kwargs.")

    except Exception as e:
        print("Could not import loader helpers from repo:", e)
        print("Falling back to printing selected config values:")
        for key in ["dim", "in_dim", "out_dim", "num_heads", "ffn_dim", "num_layers"]:
            if key in raw_cfg:
                print(f"  {key}: {raw_cfg[key]}")


if __name__ == "__main__":
    main()
