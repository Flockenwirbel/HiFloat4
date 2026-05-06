import gc
import json
import os
from typing import Any, Dict, Set, Tuple, cast

import torch
import torch.nn as nn


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _is_legacy_wan_config(config: Dict[str, Any]) -> bool:
    if "_diffusers_version" in config:
        return False
    legacy_keys = {"in_dim", "out_dim", "num_heads", "dim"}
    return any(k in config for k in legacy_keys)


def _load_wan_weight_index(load_dir: str) -> Dict[str, str]:
    index_path = os.path.join(load_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Missing shard index file: {index_path}. "
            "Legacy Wan loading requires sharded safetensors with an index JSON."
        )
    index_data = _read_json(index_path)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight_map in {index_path}")
    return weight_map


def _get_safetensor_shape(load_dir: str, weight_map: Dict[str, str], key: str) -> Tuple[int, ...]:
    from safetensors import safe_open

    shard_name = weight_map.get(key)
    if shard_name is None:
        raise KeyError(f"Key '{key}' not found in safetensors index")
    shard_path = os.path.join(load_dir, shard_name)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"Key '{key}' is not present in shard {shard_name}")
        return tuple(int(x) for x in f.get_slice(key).get_shape())


def _build_legacy_wan_init_kwargs(load_dir: str,
                                  legacy_cfg: Dict[str, Any],
                                  weight_map: Dict[str, str]) -> Dict[str, Any]:
    patch_shape = _get_safetensor_shape(load_dir, weight_map, "patch_embedding.weight")
    text_shape = _get_safetensor_shape(load_dir, weight_map, "text_embedding.0.weight")

    inner_dim = int(legacy_cfg.get("dim", patch_shape[0]))
    num_heads = int(legacy_cfg.get("num_heads", 40))
    if inner_dim % num_heads != 0:
        raise ValueError(
            f"Invalid legacy config: dim={inner_dim} is not divisible by num_heads={num_heads}"
        )

    init_kwargs: Dict[str, Any] = {
        "patch_size": tuple(int(x) for x in patch_shape[-3:]),
        "num_attention_heads": num_heads,
        "attention_head_dim": inner_dim // num_heads,
        "in_channels": int(legacy_cfg.get("in_dim", patch_shape[1])),
        "out_channels": int(legacy_cfg.get("out_dim", 16)),
        "text_dim": int(text_shape[1]),
        "freq_dim": int(legacy_cfg.get("freq_dim", 256)),
        "ffn_dim": int(legacy_cfg.get("ffn_dim", 13824)),
        "num_layers": int(legacy_cfg.get("num_layers", 40)),
        "cross_attn_norm": bool(legacy_cfg.get("cross_attn_norm", True)),
        "qk_norm": legacy_cfg.get("qk_norm", "rms_norm_across_heads"),
        "eps": float(legacy_cfg.get("eps", 1e-6)),
        "rope_max_seq_len": int(legacy_cfg.get("rope_max_seq_len", 1024)),
    }

    if "img_emb.proj.0.weight" in weight_map:
        image_shape = _get_safetensor_shape(load_dir, weight_map, "img_emb.proj.0.weight")
        init_kwargs["image_dim"] = int(image_shape[0])

    if "blocks.0.cross_attn.k_img.weight" in weight_map:
        added_k_shape = _get_safetensor_shape(load_dir, weight_map, "blocks.0.cross_attn.k_img.weight")
        init_kwargs["added_kv_proj_dim"] = int(added_k_shape[1])

    return init_kwargs


def _load_legacy_wan_weights(model: nn.Module, load_dir: str, weight_map: Dict[str, str]) -> None:
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    shard_files = sorted(set(weight_map.values()))
    expected_keys: Set[str] = set(model.state_dict().keys())
    loaded_keys: Set[str] = set()

    print(f"[Loading] Legacy Wan checkpoint detected, converting {len(shard_files)} shards...")
    for idx, shard_name in enumerate(shard_files, start=1):
        shard_path = os.path.join(load_dir, shard_name)
        print(f"  [{idx}/{len(shard_files)}] {shard_name}")

        shard_state = load_file(shard_path)
        converted_state = convert_wan_transformer_to_diffusers(dict(shard_state))
        load_result = model.load_state_dict(converted_state, strict=False)

        if load_result.unexpected_keys:
            bad = ", ".join(load_result.unexpected_keys[:5])
            raise ValueError(f"Unexpected keys after conversion: {bad}")

        loaded_keys.update(converted_state.keys())
        del shard_state
        del converted_state
        gc.collect()

    missing = sorted(expected_keys - loaded_keys)
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(
            f"Legacy checkpoint conversion incomplete: {len(missing)} model keys were not loaded. "
            f"Example missing keys: {sample}"
        )

    extra = sorted(loaded_keys - expected_keys)
    if extra:
        sample = ", ".join(extra[:10])
        raise ValueError(
            f"Legacy checkpoint conversion produced {len(extra)} unexpected keys. "
            f"Example: {sample}"
        )

    print(f"[Loading] Legacy conversion complete: loaded {len(loaded_keys)} parameters")


def _load_wan_transformer_with_fallback(load_dir: str, dtype: torch.dtype) -> nn.Module:
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    raw_cfg = _read_json(os.path.join(load_dir, "config.json"))

    # Skip from_pretrained attempt - go directly to legacy loading
    # This avoids creating a model with wrong config first
    weight_map = _load_wan_weight_index(load_dir)
    init_kwargs = _build_legacy_wan_init_kwargs(load_dir, raw_cfg, weight_map)
    
    print(f"[Loading] Creating WanTransformer3DModel with config:")
    for k, v in init_kwargs.items():
        print(f"  {k}: {v}")
    
    model = WanTransformer3DModel(**init_kwargs)
    _load_legacy_wan_weights(model, load_dir, weight_map)
    return model


def _resolve_wan_load_dir(model_path: str) -> str:
    high_noise_dir = os.path.join(model_path, "high_noise_model")
    if os.path.isdir(high_noise_dir) and os.path.isfile(os.path.join(high_noise_dir, "config.json")):
        return high_noise_dir
    return model_path


def load_wan_model(model_path: str, device: str = "cuda",
                   dtype: torch.dtype = torch.bfloat16,
                   subdir: str = None) -> nn.Module:
    if subdir:
        load_dir = os.path.join(model_path, subdir)
    else:
        load_dir = _resolve_wan_load_dir(model_path)

    print(f"[Loading] Wan 2.2 transformer from {load_dir} ...")
    config_path = os.path.join(load_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    model = _load_wan_transformer_with_fallback(load_dir, dtype)

    print(f"  Loaded: {type(model).__name__}")

    model = nn.Module.to(model, device=device, dtype=dtype)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    linear_params = sum(
        p.numel() for m in model.modules()
        if isinstance(m, nn.Linear) for p in m.parameters()
    )
    print(f"  Transformer loaded: {total_params/1e9:.2f}B total, "
          f"{linear_params/1e9:.2f}B in Linear layers")
    return model
