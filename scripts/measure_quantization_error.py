#!/usr/bin/env python3
"""
L2 Precision Comparison: BF16 vs Quantized Transformer

Measures per-layer activation error between BF16 baseline and quantized model.
This isolates the quantization error from pipeline configuration issues.

Usage:
    python scripts/measure_quantization_error.py \
        --model-path /home/dataset/Wan2.2-I2V-A14B \
        --quantized-path ./quantized_wan_output

Output:
    - Console table: per-layer MSE, cosine similarity, max absolute error
    - JSON report: scripts/quantization_error_report.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))
sys.path.insert(0, _SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="L2 precision comparison: BF16 vs Quantized")
    p.add_argument("--model-path", type=str, default="/home/dataset/Wan2.2-I2V-A14B",
                   help="Path to original Wan 2.2 model directory")
    p.add_argument("--quantized-path", type=str, default="./quantized_wan_output",
                   help="Path to quantized transformer checkpoint dir")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-samples", type=int, default=3,
                   help="Number of random inputs to average over")
    p.add_argument("--output-file", type=str, default=None,
                   help="Output JSON report path (default: scripts/quantization_error_report.json)")
    return p.parse_args()


def compute_metrics(tensor_a, tensor_b):
    """Compute error metrics between two tensors."""
    a = tensor_a.detach().float().flatten()
    b = tensor_b.detach().float().flatten()

    diff = a - b
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_ae = diff.abs().max().item()

    # Cosine similarity
    norm_a = a.norm().item()
    norm_b = b.norm().item()
    if norm_a > 1e-8 and norm_b > 1e-8:
        cos_sim = (a @ b).item() / (norm_a * norm_b)
    else:
        cos_sim = 1.0

    # Signal-to-noise ratio (dB)
    if mse > 1e-12:
        signal_power = (a ** 2).mean().item()
        snr = 10.0 * (signal_power / mse) ** 0.5  # sqrt for amplitude SNR
    else:
        snr = float('inf')

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_ae,
        "cosine_similarity": cos_sim,
        "snr_db": snr if snr != float('inf') else 999.0,
    }


def register_output_hooks(model, storage_dict, prefix=""):
    """Register forward hooks to capture layer outputs."""
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                storage_dict[name] = output.detach().clone()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    storage_dict[name] = output[0].detach().clone()
        return hook_fn

    for name, module in model.named_modules():
        # Hook the main transformer blocks
        if isinstance(module, nn.Linear):
            full_name = f"{prefix}{name}" if prefix else name
            hooks.append(module.register_forward_hook(make_hook(full_name)))

    return hooks


def create_dummy_input(model, device, batch_size=1):
    """Create dummy input matching the model's expected shapes."""
    # Wan 2.2 I2V transformer expects:
    # - hidden_states: [B, seq_len, in_channels]
    # - encoder_hidden_states: [B, text_seq_len, text_dim]
    # - timestep: [B]
    # - (optional) image embeddings

    config = model.config if hasattr(model, 'config') else None
    in_channels = getattr(config, 'in_channels', 36) if config else 36
    text_dim = getattr(config, 'text_dim', 4096) if config else 4096

    # Spatial dimensions for 720p: 720/2/2 = 180, 1280/2/2 = 320 (patch-level)
    # But we use smaller for faster testing
    seq_len = 16 * 10  # small seq_len for speed
    text_seq_len = 64

    hidden_states = torch.randn(batch_size, seq_len, in_channels,
                                device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim,
                                        device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.bfloat16)

    kwargs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "return_dict": True,
    }

    # Add image embedding if model expects it
    if config and getattr(config, 'image_dim', None) is not None:
        image_dim = config.image_dim
        kwargs["image_embeds"] = torch.randn(batch_size, seq_len, image_dim,
                                              device=device, dtype=torch.bfloat16)

    return kwargs


def main():
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16

    output_file = args.output_file or os.path.join(
        _SCRIPT_DIR, "scripts", "quantization_error_report.json")

    print("=" * 70)
    print("L2 Precision Comparison: BF16 vs Quantized Transformer")
    print("=" * 70)

    # --- Load BF16 model ---
    print("\n[1/3] Loading BF16 transformer...")
    from hifloat4.wan_model_utils import load_wan_model
    bf16_model = load_wan_model(args.model_path, device=device, dtype=dtype)
    bf16_model.eval()
    print(f"  BF16 model: {sum(p.numel() for p in bf16_model.parameters())/1e9:.2f}B params")

    # --- Load Quantized model ---
    print("\n[2/3] Loading Quantized transformer...")
    from hifloat4.wan_video_pipeline import _load_quantized_transformer
    quant_model = _load_quantized_transformer(
        args.model_path, args.quantized_path, device=device, dtype=dtype,
        subdir="high_noise_model")
    quant_model.eval()
    print(f"  Quantized model: {sum(p.numel() for p in quant_model.parameters())/1e9:.2f}B params")

    # --- Compare ---
    print(f"\n[3/3] Running {args.num_samples} forward passes...")

    # Storage for hooks
    bf16_outputs = {}
    quant_outputs = {}

    # Register hooks (only on transformer blocks for manageable output)
    print("  Registering forward hooks on linear layers...")
    bf16_hooks = register_output_hooks(bf16_model, bf16_outputs, prefix="bf16.")
    quant_hooks = register_output_hooks(quant_model, quant_outputs, prefix="quant.")

    # Accumulate metrics across samples
    all_layer_metrics = {}

    for sample_idx in range(args.num_samples):
        bf16_outputs.clear()
        quant_outputs.clear()

        # Create same input for both
        dummy_input = create_dummy_input(bf16_model, device)

        print(f"  Sample {sample_idx+1}/{args.num_samples}: "
              f"hidden_states shape={dummy_input['hidden_states'].shape}")

        # Forward pass BF16
        with torch.no_grad():
            t0 = time.time()
            _ = bf16_model(**dummy_input)
            bf16_time = time.time() - t0

        # Forward pass Quantized
        with torch.no_grad():
            t0 = time.time()
            _ = quant_model(**dummy_input)
            quant_time = time.time() - t0

        print(f"    BF16: {bf16_time:.2f}s, Quantized: {quant_time:.2f}s")

        # Compare matching layers
        bf16_names = set(bf16_outputs.keys())
        quant_names = set(quant_outputs.keys())
        common = sorted(bf16_names & quant_names)

        if sample_idx == 0:
            print(f"    Hooked layers: BF16={len(bf16_names)}, "
                  f"Quant={len(quant_names)}, Common={len(common)}")

        for name in common:
            b_name = f"bf16.{name}" if not name.startswith("bf16.") else name
            q_name = f"quant.{name}" if not name.startswith("quant.") else name

            # Strip prefix for matching
            layer_key = name.split(".", 1)[1] if "." in name else name

            b_out = bf16_outputs.get(b_name)
            q_out = quant_outputs.get(q_name)

            if b_out is None or q_out is None:
                continue

            # Handle shape mismatch (e.g., due to MoE)
            if b_out.shape != q_out.shape:
                min_shape = [min(a, b) for a, b in zip(b_out.shape, q_out.shape)]
                slices = tuple(slice(0, s) for s in min_shape)
                b_out = b_out[slices]
                q_out = q_out[slices]

            try:
                metrics = compute_metrics(b_out, q_out)
                if layer_key not in all_layer_metrics:
                    all_layer_metrics[layer_key] = {k: [] for k in metrics}
                for k, v in metrics.items():
                    all_layer_metrics[layer_key][k].append(v)
            except Exception as e:
                pass  # Skip layers with computation errors

    # Remove hooks
    for h in bf16_hooks + quant_hooks:
        h.remove()

    # --- Aggregate results ---
    print("\n" + "=" * 70)
    print("QUANTIZATION ERROR REPORT")
    print("=" * 70)

    report = {
        "model_path": args.model_path,
        "quantized_path": args.quantized_path,
        "num_samples": args.num_samples,
        "layers": {},
        "summary": {},
    }

    # Average metrics per layer
    layer_summaries = []
    for layer_key in sorted(all_layer_metrics.keys()):
        metrics = all_layer_metrics[layer_key]
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        report["layers"][layer_key] = avg_metrics
        layer_summaries.append((layer_key, avg_metrics))

    # Sort by MSE (worst first)
    layer_summaries.sort(key=lambda x: x[1]["mse"], reverse=True)

    # Print top-20 worst layers
    print(f"\nTop 20 layers with highest MSE (quantization error):")
    print(f"{'Layer':<60} {'MSE':>12} {'MAE':>12} {'MaxErr':>12} {'CosSim':>8} {'SNR':>8}")
    print("-" * 120)

    total_mse = 0
    total_cos = 0
    for i, (name, m) in enumerate(layer_summaries[:20]):
        print(f"{name:<60} {m['mse']:>12.6f} {m['mae']:>12.6f} "
              f"{m['max_abs_error']:>12.6f} {m['cosine_similarity']:>8.4f} {m['snr_db']:>8.2f}")

    # Print summary
    for name, m in layer_summaries:
        total_mse += m["mse"]
        total_cos += m["cosine_similarity"]

    n_layers = len(layer_summaries)
    avg_mse = total_mse / n_layers if n_layers > 0 else 0
    avg_cos = total_cos / n_layers if n_layers > 0 else 1

    print(f"\n{'SUMMARY':<60} {'MSE':>12} {'CosSim':>8}")
    print("-" * 80)
    print(f"{'Average over ' + str(n_layers) + ' layers':<60} {avg_mse:>12.6f} {avg_cos:>8.4f}")

    # Find worst 5% layers
    n_worst = max(1, n_layers // 20)
    worst_layers = [name for name, _ in layer_summaries[:n_worst]]
    print(f"\nWorst {n_worst} layers (candidates for high_precision_layers):")
    for name in worst_layers:
        m = report["layers"][name]
        print(f"  {name}: MSE={m['mse']:.6f}, CosSim={m['cosine_similarity']:.4f}")

    report["summary"] = {
        "total_layers_compared": n_layers,
        "avg_mse": avg_mse,
        "avg_cosine_similarity": avg_cos,
        "worst_layers": worst_layers,
        "worst_5pct_mse": sum(report["layers"][n]["mse"] for n in worst_layers) / len(worst_layers),
    }

    # Save report
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_file}")

    # Cleanup
    del bf16_model, quant_model
    torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()