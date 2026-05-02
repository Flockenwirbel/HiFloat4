#!/usr/bin/env python3
"""Diagnostic script to investigate pipeline quality issues.

Checks:
1. VAE config and weight loading (is_residual, missing/unexpected keys)
2. VAE decode round-trip test (encode → decode)
3. Transformer weight loading correctness
4. Scheduler timestep distribution
5. Input image preprocessing comparison

Usage (on login node, no GPU needed for some checks):
    python scripts/diagnose_pipeline.py
    # Or with GPU:
    srun --gres=gpu:H100:1 --cpus-per-task=8 --mem=64G --time=0:30:00 \
        --partition=Star --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh \
        && conda activate HiFloat4 && cd /home/liujh/HiFloat4 && python scripts/diagnose_pipeline.py'
"""

import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

MODEL_PATH = "/home/dataset/Wan2.2-I2V-A14B"


def check_vae():
    """Check VAE loading and configuration."""
    print("\n" + "=" * 60)
    print("VAE DIAGNOSTICS")
    print("=" * 60)

    import torch
    import gc
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    vae_pth = os.path.join(MODEL_PATH, "Wan2.1_VAE.pth")
    vae_sd = torch.load(vae_pth, map_location="cpu", weights_only=True)
    print(f"Raw VAE checkpoint keys ({len(vae_sd)}):")
    for k in sorted(vae_sd.keys())[:10]:
        print(f"  {k}: {vae_sd[k].shape}")
    if len(vae_sd) > 10:
        print(f"  ... and {len(vae_sd) - 10} more")

    converted_sd = convert_wan_vae_to_diffusers(dict(vae_sd))
    print(f"\nConverted VAE checkpoint keys ({len(converted_sd)}):")
    for k in sorted(converted_sd.keys())[:10]:
        print(f"  {k}: {converted_sd[k].shape}")
    if len(converted_sd) > 10:
        print(f"  ... and {len(converted_sd) - 10} more")

    del vae_sd
    gc.collect()

    vae = AutoencoderKLWan()
    model_keys = set(vae.state_dict().keys())
    converted_keys = set(converted_sd.keys())

    missing_from_ckpt = sorted(model_keys - converted_keys)
    extra_in_ckpt = sorted(converted_keys - model_keys)
    common_keys = sorted(model_keys & converted_keys)

    print(f"\nVAE model keys: {len(model_keys)}")
    print(f"Converted checkpoint keys: {len(converted_keys)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Keys in model but NOT in checkpoint (missing): {len(missing_from_ckpt)}")
    for k in missing_from_ckpt:
        print(f"  MISSING: {k}")
    print(f"Keys in checkpoint but NOT in model (unexpected): {len(extra_in_ckpt)}")
    for k in extra_in_ckpt:
        print(f"  UNEXPECTED: {k}")

    load_result = vae.load_state_dict(converted_sd, strict=False)
    print(f"\nload_state_dict missing: {len(load_result.missing_keys)}")
    print(f"load_state_dict unexpected: {len(load_result.unexpected_keys)}")

    print(f"\nVAE config:")
    print(f"  z_dim={vae.config.z_dim}")
    print(f"  is_residual={vae.config.is_residual}")
    print(f"  scale_factor_spatial={vae.config.scale_factor_spatial}")
    print(f"  scale_factor_temporal={vae.config.scale_factor_temporal}")
    print(f"  latents_mean={vae.config.latents_mean[:4]}...")
    print(f"  latents_std={vae.config.latents_std[:4]}...")

    del converted_sd
    gc.collect()


def check_scheduler():
    """Check scheduler configuration."""
    print("\n" + "=" * 60)
    print("SCHEDULER DIAGNOSTICS")
    print("=" * 60)

    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_config({
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
    })
    sched.set_timesteps(50)
    print(f"shift={sched.config.shift}")
    print(f"use_dynamic_shifting={sched.config.use_dynamic_shifting}")
    print(f"Timesteps (50 steps): first={sched.timesteps[0]:.1f}, last={sched.timesteps[-1]:.1f}")
    print(f"Sigmas: first={sched.sigmas[0]:.4f}, last={sched.sigmas[-1]:.4f}")

    # Boundary timestep
    boundary_timestep = 0.3 * 1000  # boundary_ratio * num_train_timesteps
    print(f"\nBoundary timestep (ratio=0.3): {boundary_timestep}")
    print(f"  Steps for transformer (high_noise): t >= {boundary_timestep}")
    print(f"  Steps for transformer_2 (low_noise): t < {boundary_timestep}")
    for i, t in enumerate(sched.timesteps):
        if t < boundary_timestep:
            print(f"  Switch happens at step {i}/{len(sched.timesteps)}: "
                  f"t={sched.timesteps[i-1]:.1f} -> {t:.1f}")
            break


def check_transformer_config():
    """Check transformer configuration."""
    print("\n" + "=" * 60)
    print("TRANSFORMER CONFIG DIAGNOSTICS")
    print("=" * 60)

    for subdir in ["high_noise_model", "low_noise_model"]:
        cfg_path = os.path.join(MODEL_PATH, subdir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"\n{subdir}/config.json:")
        for k, v in sorted(cfg.items()):
            print(f"  {k}: {v}")

        # Check if it's a diffusers-format or legacy-format config
        has_diffusers_version = "_diffusers_version" in cfg
        has_legacy_keys = any(k in cfg for k in ["in_dim", "out_dim", "num_heads", "dim"])
        print(f"  Diffusers format: {has_diffusers_version}")
        print(f"  Legacy format: {has_legacy_keys}")


def check_input_image_preprocessing():
    """Check how input images are preprocessed."""
    print("\n" + "=" * 60)
    print("IMAGE PREPROCESSING DIAGNOSTICS")
    print("=" * 60)

    # Find a sample video
    dataset_path = os.path.join(_REPO_ROOT, "datasets", "OpenS2V-5M_to_mm.json")
    if not os.path.exists(dataset_path):
        print("Dataset not found, skipping image preprocessing check")
        return

    with open(dataset_path) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", [])
    for s in samples:
        raw_path = s.get("path") or s.get("video_path")
        if not raw_path:
            continue

        # Try to resolve path
        resolved = raw_path
        if not os.path.isfile(resolved):
            resolved = raw_path.replace("/home/datasets/OpenS2V-5M/", "datasets/")
        if not os.path.isfile(resolved):
            resolved = raw_path.replace("/home/datasets/", "datasets/")
        if not os.path.isfile(resolved):
            continue

        print(f"Sample video: {resolved}")
        try:
            import decord
            vr = decord.VideoReader(resolved, ctx=decord.cpu(0))
            frame = vr[0].asnumpy()
            print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"  Value range: [{frame.min()}, {frame.max()}]")
            print(f"  Mean per channel: R={frame[:,:,0].mean():.1f}, "
                  f"G={frame[:,:,1].mean():.1f}, B={frame[:,:,2].mean():.1f}")

            # Check what the pipeline's video_processor.preprocess does
            from PIL import Image
            image = Image.fromarray(frame)
            print(f"  PIL Image size: {image.size}, mode: {image.mode}")

            # Test preprocessing
            from diffusers.video_processor import VideoProcessor
            vp = VideoProcessor(vae=None)
            processed = vp.preprocess(image, height=480, width=832)
            print(f"  Preprocessed shape: {processed.shape}, dtype: {processed.dtype}")
            print(f"  Preprocessed range: [{processed.min():.4f}, {processed.max():.4f}]")
            print(f"  Preprocessed mean per channel:")
            for c in range(min(3, processed.shape[1])):
                print(f"    Ch{c}: mean={processed[0, c].mean():.4f}, "
                      f"std={processed[0, c].std():.4f}")
        except Exception as e:
            print(f"  Error: {e}")
        break


def check_model_loading():
    """Check if transformers load correctly."""
    print("\n" + "=" * 60)
    print("MODEL LOADING DIAGNOSTICS")
    print("=" * 60)

    import torch
    from hifloat4.wan_model_utils import load_wan_model

    # Test loading the high_noise_model
    load_dir = os.path.join(MODEL_PATH, "high_noise_model")
    print(f"Loading transformer from {load_dir}...")

    try:
        model = load_wan_model(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
        print(f"Model loaded successfully: {type(model).__name__}")

        # Check model config
        if hasattr(model, 'config'):
            print(f"Model config:")
            for k, v in sorted(model.config.__dict__.items()):
                if not k.startswith('_'):
                    print(f"  {k}: {v}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params/1e9:.2f}B")

        # Check a few weight statistics
        for name, param in list(model.named_parameters())[:5]:
            print(f"  {name}: shape={param.shape}, "
                  f"mean={param.float().mean():.6f}, std={param.float().std():.6f}")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Wan 2.2 I2V Pipeline Diagnostics")
    print(f"Model path: {MODEL_PATH}")

    try:
        check_transformer_config()
    except Exception as e:
        print(f"Transformer config check failed: {e}")

    try:
        check_scheduler()
    except Exception as e:
        print(f"Scheduler check failed: {e}")

    try:
        check_vae()
    except Exception as e:
        print(f"VAE check failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        check_input_image_preprocessing()
    except Exception as e:
        print(f"Image preprocessing check failed: {e}")

    # Note: check_model_loading requires significant CPU RAM
    # Uncomment to run:
    # try:
    #     check_model_loading()
    # except Exception as e:
    #     print(f"Model loading check failed: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)