#!/bin/bash
#SBATCH --job-name=vbench_eval
#SBATCH --partition=Star
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vbench_eval_%j.out
#SBATCH --error=logs/vbench_eval_%j.err

set -e
cd /home/liujh/HiFloat4

source /home/liujh/miniconda3/etc/profile.d/conda.sh
conda activate HiFloat4

echo "=== VBench Evaluation ==="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Usage: sbatch scripts/eval_vbench.sbatch
# Evaluates videos in ./quantized_wan_output/generated_videos_v2 by default.
# To evaluate other directories, pass --videos-dir to evaluate_vbench.py directly.

VIDEOS_DIR="${1:-./quantized_wan_output/generated_videos_v2}"

python3 scripts/evaluate_vbench.py --videos-dir "$VIDEOS_DIR"

echo ""
echo "=== Evaluation complete ==="
echo "Date: $(date)"