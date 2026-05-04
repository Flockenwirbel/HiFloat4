#!/usr/bin/env python3
"""CLI entry point for Wan 2.2 I2V generation."""

import argparse
import json
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hifloat4.wan_video_pipeline import (
    build_bf16_pipeline,
    build_quantized_pipeline,
    extract_first_frame,
    load_dataset,
    save_video_frames_as_mp4,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate I2V videos with quantized Wan 2.2")
    p.add_argument("--model-path", type=str, default="/home/dataset/Wan2.2-I2V-A14B",
                   help="Path to original Wan 2.2 model directory")
    p.add_argument("--quantized-path", type=str, default="./quantized_wan_output",
                   help="Path to quantized transformer checkpoint dir")
    p.add_argument("--dataset", type=str, default="datasets/OpenS2V-5M_to_mm.json")
    p.add_argument("--output-dir", type=str, default="./generated_videos")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--num-videos", type=int, default=0,
                   help="Number of videos to generate (0 = all available)")
    p.add_argument("--num-frames", type=int, default=61,
                   help="Number of frames to generate per video (61 ~ 4s @ 16fps, competition default)")
    p.add_argument("--resolution", type=str, default="720p",
                   help="Resolution: 480p or 720p (competition requires 720p)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--bf16-baseline", action="store_true",
                   help="Run BF16 baseline (no quantization) for comparison")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    samples = load_dataset(args.dataset,
                           max_samples=args.num_videos if args.num_videos > 0 else -1)
    if not samples:
        print("[FATAL] No valid samples found. Check dataset path.")
        return

    if args.bf16_baseline:
        print("[Mode] BF16 BASELINE — no quantization")
        pipe = build_bf16_pipeline(args.model_path, args.device, dtype)
        args.output_dir = args.output_dir.rstrip('/') + '_bf16_baseline'
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        pipe = build_quantized_pipeline(args.model_path, args.quantized_path, args.device, dtype)

    resolution_map = {"480p": (480, 832), "720p": (720, 1280)}
    height, width = resolution_map.get(args.resolution, (720, 1280))

    print(f"\n[Generate] Generating {len(samples)} videos...")
    print(f"  Resolution: {height}x{width}, Frames: {args.num_frames}")
    print(f"  Guidance: {args.guidance_scale}, Steps: {args.num_inference_steps}")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    generated = 0

    for idx, sample in enumerate(samples):
        video_path = sample.get("resolved_path") or sample.get("path") or sample.get("video_path")
        caption = sample.get("cap", sample.get("caption", ""))
        sample_id = sample.get("id", f"sample_{idx:04d}")

        if not video_path or not caption:
            print(f"  [{idx+1}/{len(samples)}] Skipping (missing path/caption)")
            continue

        first_frame_orig = extract_first_frame(video_path)
        if first_frame_orig is None:
            continue

        from PIL import Image
        image = Image.fromarray(first_frame_orig)

        output_path = os.path.join(args.output_dir, f"{sample_id}.mp4")

        print(f"  [{idx+1}/{len(samples)}] Generating: {sample_id}")
        print(f"    Caption: {caption[:80]}...")

        try:
            torch.manual_seed(args.seed + idx)
            result = pipe(
                image=image,
                prompt=caption,
                negative_prompt="",
                height=height,
                width=width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

            video_frames = result.frames[0]

            # Safety net: Replace first frame with input image to ensure pixel-perfect
            # consistency. With expand_timesteps=True the first frame latent is already
            # protected during denoising, but VAE encode/decode may still introduce
            # minor color shifts. Keeping this replacement as a safeguard.
            input_array = np.array(image.resize((width, height)))
            if input_array.shape[-1] == 3:  # RGB
                video_frames[0] = input_array / 255.0  # Normalize to [0, 1]

            save_video_frames_as_mp4(video_frames, output_path, fps=args.fps)
            generated += 1
            print(f"    Saved: {output_path} ({len(video_frames)} frames)")

        except Exception as e:
            print(f"    [Error] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[Done] Generated {generated}/{len(samples)} videos")
    print(f"  Output dir: {args.output_dir}")

    meta = {
        "num_generated": generated,
        "total_samples": len(samples),
        "model_path": args.model_path,
        "quantized_path": args.quantized_path,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "fps": args.fps,
    }
    meta_path = os.path.join(args.output_dir, "generation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
