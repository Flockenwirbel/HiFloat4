#!/bin/bash
#SBATCH --job-name=wan_quant
#SBATCH --partition=Star
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/quantize_%j.out
#SBATCH --error=logs/quantize_%j.err

set -e
cd /home/liujh/HiFloat4

source /home/liujh/miniconda3/etc/profile.d/conda.sh
conda activate HiFloat4

echo "=== Wan 2.2 W4A4 Quantization ==="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Usage:
#   Default (hifx4, rotation, sensitivity):  sbatch scripts/quantize.sbatch
#   Custom quant type:                       sbatch scripts/quantize.sbatch --quant-type mxfp4
#   All args forwarded to quantize_wan.py

python3 quantize_wan.py \
    --model-path /home/dataset/Wan2.2-I2V-A14B \
    --output-dir ./quantized_wan_output \
    --quant-type hifx4 \
    --max-high-precision-layers 2 \
    --rotation-mode pad \
    --rotation-seed 42 \
    "$@"

echo ""
echo "=== Quantization complete ==="
echo "Date: $(date)"