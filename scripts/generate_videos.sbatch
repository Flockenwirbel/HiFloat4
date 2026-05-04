#!/bin/bash
#SBATCH --job-name=wan_gen
#SBATCH --partition=Star
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/generate_videos_%j.out
#SBATCH --error=logs/generate_videos_%j.err

set -e
cd /home/liujh/HiFloat4

source /home/liujh/miniconda3/etc/profile.d/conda.sh
conda activate HiFloat4

echo "=== Wan 2.2 I2V Video Generation ==="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Usage:
#   Quantized (default):    sbatch scripts/generate_videos.sbatch
#   BF16 baseline:          sbatch scripts/generate_videos.sbatch --bf16-baseline
#   Custom number of videos: sbatch scripts/generate_videos.sbatch --num-videos 5
#   All args are forwarded to generate_videos.py

python3 generate_videos.py \
    --model-path /home/dataset/Wan2.2-I2V-A14B \
    --quantized-path ./quantized_wan_output \
    --dataset datasets/OpenS2V-5M_to_mm.json \
    --output-dir ./quantized_wan_output/generated_videos_v2 \
    --resolution 720p \
    --num-frames 61 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --fps 16 \
    --seed 42 \
    "$@"

echo ""
echo "=== Generation complete ==="
echo "Date: $(date)"