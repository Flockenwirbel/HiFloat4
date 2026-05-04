#!/usr/bin/env python3
"""
VBench evaluation for quantized and BF16 baseline videos.
Evaluates 5 dimensions: imaging_quality, aesthetic_quality, overall_quality,
                         subject_consistency, motion_smoothness

Usage:
    python scripts/evaluate_vbench.py --videos-dir ./quantized_wan_output/generated_videos_v2
    python scripts/evaluate_vbench.py --videos-dir ./quantized_wan_output/generated_videos_bf16_baseline
"""

import argparse
import json
import os
import sys


# The 5 dimensions requested
DIMENSIONS = [
    "imaging_quality",       # pyiqa musiq_spaq ✅
    "aesthetic_quality",     # CLIP ViT-L/14 ✅
    "overall_consistency",   # CLIP ViT-B/32 (text-video alignment) ✅
    "subject_consistency",   # DINO ✅
    "motion_smoothness",     # AMT-S ✅
]


def main():
    parser = argparse.ArgumentParser(description="VBench evaluation (5 metrics)")
    parser.add_argument("--videos-dir", type=str, required=True,
                        help="Directory containing .mp4 videos to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: <videos-dir>/vbench_eval)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dimensions", type=str, default=None,
                        help="Comma-separated list of dimensions to evaluate (default: 5 standard dims)")
    args = parser.parse_args()

    from vbench import VBench

    videos_dir = os.path.abspath(args.videos_dir)
    if not os.path.isdir(videos_dir):
        print(f"Error: videos directory not found: {videos_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(videos_dir, "vbench_eval")
    os.makedirs(output_dir, exist_ok=True)

    full_info_dir = os.path.join(
        os.path.dirname(__import__('vbench').__file__),
        "VBench_full_info.json"
    )

    # Initialize VBench
    my_VBench = VBench(device=args.device, full_info_dir=full_info_dir, output_path=output_dir)

    # Determine dimensions
    if args.dimensions:
        dimension_list = [d.strip() for d in args.dimensions.split(",")]
    else:
        dimension_list = DIMENSIONS

    # List video files
    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} videos in {videos_dir}")
    for vf in video_files:
        print(f"  {vf}")

    # Use a descriptive name for the evaluation run
    eval_name = os.path.basename(videos_dir)

    print(f"\n=== VBench Evaluation ===")
    print(f"Videos dir: {videos_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Dimensions: {dimension_list}")
    print(f"Evaluating as: {eval_name}")
    print(f"Mode: custom_input (supports arbitrary video files)")
    print()

    # Use mode='custom_input' so VBench processes arbitrary video files
    # It will extract prompts from filenames (or use empty prompts for non-text dims)
    my_VBench.evaluate(
        videos_path=videos_dir,
        name=eval_name,
        prompt_list=[],  # no prompts needed for quality/consistency/smoothness metrics
        dimension_list=dimension_list,
        local=True,          # use cached models, don't download
        read_frame=False,
        mode="custom_input",  # key fix: use custom_input mode for arbitrary videos
    )

    # Print results
    results_path = os.path.join(output_dir, f"{eval_name}_eval_results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n{'='*60}")
        print(f"VBench Results for: {eval_name}")
        print(f"{'='*60}")
        for dim, dim_results in results.items():
            if isinstance(dim_results, dict):
                for k, v in dim_results.items():
                    if k not in ('video_path', 'prompt'):
                        print(f"  {dim}/{k}: {v}")
            else:
                print(f"  {dim}: {dim_results}")
        print()
        print(f"Full results saved to: {results_path}")
    else:
        # Try to find any results file
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(".json"):
                fp = os.path.join(output_dir, f)
                print(f"\nResults file: {fp}")
                with open(fp) as fh:
                    print(json.dumps(json.load(fh), indent=2))

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()