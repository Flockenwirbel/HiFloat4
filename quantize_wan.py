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
from typing import Dict, List, Optional, Tuple

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
from hifloat4.wan_model_utils import load_wan_model
from hifloat4.wan_rotation import rotate_linear_weights


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

def _quantize_single_transformer(
    model_path: str,
    subdir: str,
    output_dir: str,
    quant_type: str,
    max_high_precision: int,
    apply_rotation: bool,
    rotation_mode: str,
    rotation_seed: int,
    skip_sensitivity: bool,
    explicit_layers: Optional[List[str]],
    device: str,
    dtype: torch.dtype,
) -> None:
    """Quantize a single transformer sub-model (high_noise_model or low_noise_model)."""
    load_dir = os.path.join(model_path, subdir)
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(
            f"Sub-model directory not found: {load_dir}")

    subdir_out = os.path.join(output_dir, subdir)
    print()
    print("#" * 60)
    print(f"  Quantizing: {subdir}")
    print(f"  Load from  : {load_dir}")
    print(f"  Save to    : {subdir_out}")
    print("#" * 60)

    # Load transformer from the specific subdirectory
    transformer = load_wan_model(model_path, device=device, dtype=dtype,
                                  subdir=subdir)

    t0 = time.time()
    transformer = quantize_wan_transformer(
        transformer,
        quant_type=quant_type,
        high_precision_layers=explicit_layers,
        apply_rotation=apply_rotation,
        rotation_mode=rotation_mode,
        rotation_seed=rotation_seed,
        calibration_input=None,
        skip_sensitivity=skip_sensitivity,
        max_high_precision=max_high_precision,
    )
    elapsed = time.time() - t0
    print(f"[Timing] {subdir} quantization completed in {elapsed:.1f}s")

    # Collect the actual high-precision layer names from the quantized model
    # (layers still using nn.Linear after replace_linear).
    # IMPORTANT: use `type() is` not `isinstance` because QLinear inherits from nn.Linear.
    high_prec_actual = [
        name for name, mod in transformer.named_modules()
        if type(mod) is nn.Linear
    ]

    metadata = {
        "model_path": model_path,
        "subdir": subdir,
        "quant_type": quant_type,
        "max_high_precision_layers": max_high_precision,
        "high_precision_layers": high_prec_actual,
        "rotation_mode": rotation_mode if apply_rotation else "none",
        "rotation_seed": rotation_seed,
        "device": device,
        "dtype": str(dtype),
    }
    save_quantized_model(transformer, subdir_out, metadata)

    del transformer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[Done] {subdir} saved to {subdir_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="W4A4 Quantization for Wan 2.2 I2V-A14B")
    parser.add_argument("--model-path", type=str, default="/home/dataset/Wan2.2-I2V-A14B")
    parser.add_argument("--output-dir", type=str, default="./quantized_wan_output")
    parser.add_argument("--quant-type", type=str, default="hifx4",
                        choices=["hifx4", "mxfp4"])
    parser.add_argument("--subdir", type=str, default="both",
                        choices=["high_noise_model", "low_noise_model", "both"],
                        help="Which transformer(s) to quantize. 'both' quantizes "
                             "high_noise_model and low_noise_model separately.")
    parser.add_argument("--max-high-precision-layers", type=int, default=2)
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
    # Quality-first defaults: enable sensitivity analysis and Hadamard rotation.
    # Previous defaults (no rotation, 0 high-precision layers) caused catastrophic
    # quality loss — see https://github.com/Flockenwirbel/HiFloat4/issues/XXX
    parser.set_defaults(no_rotation=False, skip_sensitivity=False)
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

    # Determine which subdirs to quantize
    if args.subdir == "both":
        subdirs = ["high_noise_model", "low_noise_model"]
    else:
        subdirs = [args.subdir]

    print("=" * 60)
    print("W4A4 Quantization — Wan 2.2 I2V-A14B")
    print("=" * 60)
    print(f"  Quant type      : {args.quant_type}")
    print(f"  Sub-model(s)    : {', '.join(subdirs)}")
    print(f"  High precision  : ≤{args.max_high_precision_layers} layers")
    print(f"  Rotation        : {'off' if args.no_rotation else args.rotation_mode}")
    print(f"  Sensitivity     : {'skip' if args.skip_sensitivity else 'auto'}")
    print(f"  Device          : {args.device}")
    print(f"  Dtype           : {args.dtype}")
    print(f"  Output dir      : {args.output_dir}")
    print()

    if args.load_only:
        for subdir in subdirs:
            load_dir = os.path.join(args.model_path, subdir)
            _ = load_wan_model(args.model_path, device=args.device, dtype=dtype,
                               subdir=subdir)
            print(f"[Done] Load-only check passed for {subdir}.")
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

    overall_t0 = time.time()
    for subdir in subdirs:
        _quantize_single_transformer(
            model_path=args.model_path,
            subdir=subdir,
            output_dir=args.output_dir,
            quant_type=args.quant_type,
            max_high_precision=args.max_high_precision_layers,
            apply_rotation=not args.no_rotation,
            rotation_mode=args.rotation_mode,
            rotation_seed=args.rotation_seed,
            skip_sensitivity=args.skip_sensitivity,
            explicit_layers=explicit_layers,
            device=args.device,
            dtype=dtype,
        )

    total_elapsed = time.time() - overall_t0
    print()
    print("=" * 60)
    print(f"Quantization complete! ({total_elapsed:.1f}s total)")
    print(f"  Output: {args.output_dir}/")
    for subdir in subdirs:
        print(f"    {subdir}/quantized_transformer.pt")
        print(f"    {subdir}/quantization_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
