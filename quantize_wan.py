#!/usr/bin/env python3
"""
W4A4 Quantization Pipeline for Wan 2.2 (I2V-A14B)
===================================================
Quantizes all nn.Linear layers in the Wan transformer to W4A4 using
HiF4 or MXFP4 numerical formats, with optional Hadamard rotation and
selective high-precision layers.

Usage:
    python quantize_wan.py \
        --model-path /home/dataset/Wan2.2-I2V-A14B \
        --output-dir ./quantized_wan_output \
        --quant-type hifx4 \
        --max-high-precision-layers 2 \
        --device cuda --dtype bfloat16
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import torch
import torch.nn as nn

# Add repo root to path for HiF4 imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hif4_gpu.quant_cy.utils.utils import replace_linear  # noqa: E402
from hifloat4.wan_model_utils import (
    _build_legacy_wan_init_kwargs,
    _is_legacy_wan_config,
    _load_legacy_wan_weights,
    _load_wan_transformer_with_fallback,
    _load_wan_weight_index,
    _read_json,
    _resolve_wan_load_dir,
    load_wan_model,
)
from hifloat4.wan_rotation import (
    block_diagonal_hadamard,
    fwht,
    hadamard_matrix,
    register_rotation_hooks,
    rotate_linear_weights,
)


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def compute_layer_sensitivity(model: nn.Module,
                               quant_type: str = "hifx4") -> List[Tuple[str, float]]:
    """
    Compute per-layer weight-quantization sensitivity.
    Returns list of (layer_name, relative_error) sorted by error descending.
    """
    from hif4_gpu.quant_cy.base.QType import QType
    from hif4_gpu.quant_cy.base.QTensor import quant_dequant_float

    model.eval()
    errors: List[Tuple[str, float]] = []

    with torch.no_grad():
        qtype_w = QType(quant_type).dim(0)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w_orig = module.weight.data.clone()
            w_q = quant_dequant_float(module.weight.data, qtype_w,
                                       force_py=True, force_fp32=True)
            rel_err = (w_q.float() - w_orig.float()).norm() / (w_orig.float().norm() + 1e-8)
            errors.append((name, rel_err.item()))
            module.weight.data.copy_(w_orig)

    errors.sort(key=lambda x: x[1], reverse=True)
    return errors


def select_high_precision_layers(model: nn.Module,
                                  quant_type: str,
                                  max_high_precision: int) -> List[str]:
    """Select top-k most sensitive layers to keep at high precision."""
    if max_high_precision <= 0:
        return []

    print("[Sensitivity] Computing per-layer quantization error...")
    sensitivities = compute_layer_sensitivity(model, quant_type)
    selected = [name for name, _ in sensitivities[:max_high_precision]]
    print(f"[Sensitivity] Top {max_high_precision} layers kept at high precision:")
    for name, err in sensitivities[:max_high_precision]:
        print(f"  {name}: rel_err={err:.6f}")
    return selected


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _is_legacy_wan_config(config: Dict[str, Any]) -> bool:
    # Diffusers Wan checkpoints always carry _diffusers_version.
    # Legacy single-file Wan checkpoints typically do not.
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

    # Optional I2V branches only appear in some Wan variants.
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


def _load_wan_transformer_with_fallback(
    load_dir: str,
    dtype: torch.dtype,
) -> nn.Module:
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    raw_cfg = _read_json(os.path.join(load_dir, "config.json"))

    try:
        return cast(nn.Module, WanTransformer3DModel.from_pretrained(
            load_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ))
    except ValueError as exc:
        if "expected shape" not in str(exc) and "mismatched sizes" not in str(exc):
            raise

        weight_map = _load_wan_weight_index(load_dir)
        init_kwargs = _build_legacy_wan_init_kwargs(load_dir, raw_cfg, weight_map)
        model = WanTransformer3DModel(**init_kwargs)
        _load_legacy_wan_weights(model, load_dir, weight_map)
        return model


def _resolve_wan_load_dir(model_path: str) -> str:
    high_noise_dir = os.path.join(model_path, "high_noise_model")
    if os.path.isdir(high_noise_dir) and os.path.isfile(os.path.join(high_noise_dir, "config.json")):
        return high_noise_dir
    return model_path


def load_wan_model(model_path: str, device: str = "cuda",
                   dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """
    Load the Wan 2.2 I2V-A14B transformer model.

    The model directory structure is:
      <model_path>/
        high_noise_model/  ← WanModel weights + config.json
        low_noise_model/   ← WanModel (for later denoising steps)
        models_t5_umt5-xxl-enc-bf16.pth  ← T5 encoder
        Wan2.1_VAE.pth                   ← VAE

    Since we only need to quantize the transformer, load high_noise_model/
    directly as a diffusers WanModel.
    """
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


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_wan_transformer(
    transformer: nn.Module,
    quant_type: str = "hifx4",
    high_precision_layers: Optional[List[str]] = None,
    apply_rotation: bool = True,
    rotation_mode: str = "pad",
    rotation_seed: int = 42,
    calibration_input: Optional[torch.Tensor] = None,
    skip_sensitivity: bool = False,
    max_high_precision: int = 2,
) -> nn.Module:
    """
    Apply W4A4 quantization to the Wan transformer.
    Steps: sensitivity → Hadamard rotation → replace Linear→QLinear.
    """
    high_precision_layers = list(dict.fromkeys(high_precision_layers or []))

    if max_high_precision <= 0:
        high_precision_layers = []
    elif len(high_precision_layers) > max_high_precision:
        print(f"[Config] Truncating explicit high-precision list to top {max_high_precision} entries")
        high_precision_layers = high_precision_layers[:max_high_precision]

    # Step 1: Sensitivity
    if not skip_sensitivity and max_high_precision > 0:
        auto_layers = select_high_precision_layers(
            transformer, quant_type, max_high_precision)
        # Keep explicit selections first, then fill remaining slots with auto-selected layers.
        merged = list(high_precision_layers)
        for name in auto_layers:
            if len(merged) >= max_high_precision:
                break
            if name not in merged:
                merged.append(name)
        high_precision_layers = merged

    # Step 2: Hadamard rotation
    H_dict: Dict[int, torch.Tensor] = {}
    if apply_rotation:
        print(f"[Rotation] Applying Hadamard rotation (mode={rotation_mode})...")
        H_dict = rotate_linear_weights(transformer, mode=rotation_mode, seed=rotation_seed)

    # Step 3: Replace Linear → QLinear
    print(f"[Quantization] Replacing nn.Linear → QLinear (w_Q={quant_type}, in_Q={quant_type})...")
    print(f"[Quantization] High-precision layers excluded: {len(high_precision_layers)}")
    for name in high_precision_layers:
        print(f"  Excluding: {name}")

    replace_linear(
        transformer,
        w_Q=quant_type,
        in_Q=quant_type,
        quant_grad=False,
        exclude_layers=high_precision_layers,
    )

    qlinear_count = sum(1 for m in transformer.modules() if 'QLinear' in type(m).__name__)
    linear_count = sum(1 for m in transformer.modules() if type(m) is nn.Linear)
    print(f"[Quantization] QLinear: {qlinear_count}, Linear (high prec): {linear_count}")
    return transformer


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_quantized_model(transformer: nn.Module, output_dir: str,
                          metadata: dict) -> None:
    """Save the quantized transformer state_dict and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Save] Saving quantized model to {output_dir} ...")
    checkpoint_path = os.path.join(output_dir, "quantized_transformer.pt")
    torch.save({
        'model_state_dict': transformer.state_dict(),
        'metadata': metadata,
    }, checkpoint_path)

    meta_path = os.path.join(output_dir, "quantization_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"[Save] Done. Model: {size_mb:.1f} MB | Metadata: {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="W4A4 Quantization for Wan 2.2 I2V-A14B")
    parser.add_argument("--model-path", type=str, default="/home/dataset/Wan2.2-I2V-A14B")
    parser.add_argument("--output-dir", type=str, default="./quantized_wan_output")
    parser.add_argument("--quant-type", type=str, default="hifx4",
                        choices=["hifx4", "mxfp4"])
    parser.add_argument("--max-high-precision-layers", type=int, default=0)
    parser.add_argument("--rotation-mode", type=str, default="pad",
                        choices=["pad", "block"])
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument("--no-rotation", dest="no_rotation", action="store_true")
    parser.add_argument("--enable-rotation", dest="no_rotation", action="store_false")
    parser.add_argument("--skip-sensitivity", dest="skip_sensitivity", action="store_true")
    parser.add_argument("--enable-sensitivity", dest="skip_sensitivity", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--high-precision-layers", type=str, default=None)
    parser.add_argument("--load-only", action="store_true",
                        help="Only load and validate the transformer, then exit")
    # Safer quality-first defaults: enable sensitivity, keep rotation off unless explicit.
    parser.set_defaults(no_rotation=True, skip_sensitivity=False)
    args = parser.parse_args()

    # Validate constraints
    if args.quant_type == "hifx4" and args.max_high_precision_layers > 2:
        print("[Warn] HiF4 allows max 2 high-precision layers, capping to 2")
        args.max_high_precision_layers = 2
    elif args.quant_type == "mxfp4" and args.max_high_precision_layers > 5:
        print("[Warn] MXFP4 allows max 5 high-precision layers, capping to 5")
        args.max_high_precision_layers = 5

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("W4A4 Quantization — Wan 2.2 I2V-A14B")
    print("=" * 60)
    print(f"  Quant type      : {args.quant_type}")
    print(f"  High precision  : ≤{args.max_high_precision_layers} layers")
    print(f"  Rotation        : {'off' if args.no_rotation else args.rotation_mode}")
    print(f"  Sensitivity     : {'skip' if args.skip_sensitivity else 'auto'}")
    print(f"  Device          : {args.device}")
    print(f"  Dtype           : {args.dtype}")
    print(f"  Output dir      : {args.output_dir}")
    print()

    transformer = load_wan_model(args.model_path, device=args.device, dtype=dtype)

    if args.load_only:
        print("[Done] Load-only check passed.")
        return

    explicit_layers = None
    if args.high_precision_layers:
        explicit_layers = [n.strip() for n in args.high_precision_layers.split(",")]
        print(f"[Config] Explicit high-precision layers: {explicit_layers}")
        if args.max_high_precision_layers <= 0:
            args.max_high_precision_layers = len(explicit_layers)
            print(
                "[Config] max-high-precision-layers not set; "
                f"using explicit list length={args.max_high_precision_layers}"
            )

            if args.quant_type == "hifx4" and args.max_high_precision_layers > 2:
                print("[Warn] HiF4 allows max 2 high-precision layers, capping to 2")
                args.max_high_precision_layers = 2
            elif args.quant_type == "mxfp4" and args.max_high_precision_layers > 5:
                print("[Warn] MXFP4 allows max 5 high-precision layers, capping to 5")
                args.max_high_precision_layers = 5

    t0 = time.time()
    transformer = quantize_wan_transformer(
        transformer,
        quant_type=args.quant_type,
        high_precision_layers=explicit_layers,
        apply_rotation=not args.no_rotation,
        rotation_mode=args.rotation_mode,
        rotation_seed=args.rotation_seed,
        calibration_input=None,
        skip_sensitivity=args.skip_sensitivity,
        max_high_precision=args.max_high_precision_layers,
    )
    elapsed = time.time() - t0
    print(f"[Timing] Quantization completed in {elapsed:.1f}s")

    metadata = {
        "model_path": args.model_path,
        "quant_type": args.quant_type,
        "max_high_precision_layers": args.max_high_precision_layers,
        "rotation_mode": args.rotation_mode if not args.no_rotation else "none",
        "rotation_seed": args.rotation_seed,
        "device": args.device,
        "dtype": args.dtype,
    }
    save_quantized_model(transformer, args.output_dir, metadata)

    # No need to save H_dict — FWHT reconstructs rotation from metadata (seed + dims)

    print()
    print("=" * 60)
    print("Quantization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()