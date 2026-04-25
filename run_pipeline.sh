#!/bin/bash
#
# W4A4 Quantized Wan 2.2 Pipeline — Execution Script
#
# Run directly on fuse0 — auto-requests a GPU node via srun if needed.
#
#   cd ~/HiFloat4
#   bash run_pipeline.sh
#
# Options:
#   bash run_pipeline.sh --quant-type mxfp4 --no-rotation --skip-sensitivity
#   bash run_pipeline.sh --load-only
#   bash run_pipeline.sh --gres gpu:H100:1 --min-gpu-mem 70000
#
set -e

# ============================================================
# Configuration — edit these as needed
# ============================================================
MODEL_PATH="/home/dataset/Wan2.2-I2V-A14B"
OUTPUT_DIR="./quantized_wan_output"
DATASET_PATH="datasets/OpenS2V-5M_to_mm.json"
QUANT_TYPE="hifx4"          # hifx4  or  mxfp4
MAX_HIGH_PRECISION=2         # HiF4: ≤2,  MXFP4: ≤5
ROTATION_MODE="pad"          # pad  or  block
ROTATION_SEED=42
SKIP_SENSITIVITY=false
NO_ROTATION=false
LOAD_ONLY=false

# srun resource request
SRUN_GRES="gpu:H100:1"
SRUN_CPUS="8"
SRUN_MEM="64G"
SRUN_TIME="8:00:00"
SRUN_PARTITION="Star"        # Partition name (from 'sinfo'). Empty = let SLURM choose.
SRUN_EXTRA=""                # e.g. "--constraint=a100"
MIN_GPU_MEM_MIB=70000

# Save original CLI args before parsing (needed for srun forwarding)
ORIGINAL_ARGS=("$@")

# ============================================================
# Parse CLI overrides
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --quant-type)        QUANT_TYPE="$2";        shift 2 ;;
        --no-rotation)       NO_ROTATION=true;       shift   ;;
        --skip-sensitivity)  SKIP_SENSITIVITY=true;  shift   ;;
        --gpu)               SRUN_GRES="gpu:$2";     shift 2 ;;   # e.g. --gpu 2
        --gres)              SRUN_GRES="$2";         shift 2 ;;   # e.g. --gres gpu:H100:1
        --partition)         SRUN_PARTITION="$2";    shift 2 ;;
        --constraint)        SRUN_EXTRA="--constraint=$2"; shift 2 ;;
        --mem)               SRUN_MEM="$2";          shift 2 ;;
        --time)              SRUN_TIME="$2";          shift 2 ;;
        --min-gpu-mem)       MIN_GPU_MEM_MIB="$2";   shift 2 ;;
        --load-only)         LOAD_ONLY=true;          shift   ;;
        *) echo "Unknown option: $1"; shift ;;
    esac
done

# ============================================================
# Auto-detect GPU — relaunch via srun if needed
# ============================================================
# Check: are we inside a SLURM allocation AND does nvidia-smi actually work?
GPU_AVAILABLE=false
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Inside SLURM — verify GPU actually works
    if nvidia-smi -L &>/dev/null; then
        GPU_AVAILABLE=true
    else
        echo "[Warn] SLURM_JOB_ID=${SLURM_JOB_ID} but nvidia-smi -L failed — driver may not be loaded"
    fi
else
    # Not in SLURM — try nvidia-smi anyway (maybe we're on a GPU node directly)
    if nvidia-smi -L &>/dev/null; then
        GPU_AVAILABLE=true
    fi
fi

if [ "$GPU_AVAILABLE" != "true" ]; then
    # Use configured partition (defaults to "Star" per sinfo)

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  No usable GPU detected — submitting to SLURM           ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo "  Partition : ${SRUN_PARTITION}"
    echo "  GPU       : ${SRUN_GRES}"
    echo "  CPUs      : ${SRUN_CPUS}"
    echo "  Memory    : ${SRUN_MEM}"
    echo "  Time      : ${SRUN_TIME}"
    echo ""
    exec srun \
        --gres="${SRUN_GRES}" \
        --cpus-per-task="${SRUN_CPUS}" \
        --mem="${SRUN_MEM}" \
        --time="${SRUN_TIME}" \
        ${SRUN_PARTITION:+--partition="${SRUN_PARTITION}"} \
        ${SRUN_EXTRA:+"${SRUN_EXTRA}"} \
        --pty bash "$0" "${ORIGINAL_ARGS[@]}"
    # exec replaces this shell — code below only runs inside the allocation
fi

# ============================================================
# Sanity checks (running on GPU from here)
# ============================================================
echo "============================================"
echo "W4A4 Quantized Wan 2.2 Pipeline"
echo "============================================"
echo "Job ID : ${SLURM_JOB_ID:-interactive (no srun)}"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo ""
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'UNKNOWN')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

GPU_MEM_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)
if [[ -n "$GPU_MEM_TOTAL_MIB" ]] && [[ "$GPU_MEM_TOTAL_MIB" -lt "$MIN_GPU_MEM_MIB" ]]; then
    echo "[FATAL] GPU memory is insufficient for Wan2.2-I2V-A14B quantization."
    echo "        Detected: ${GPU_MEM_TOTAL_MIB} MiB, required: >= ${MIN_GPU_MEM_MIB} MiB"
    echo "        Try running with a larger GPU, for example:"
    echo "        bash run_pipeline.sh --gres gpu:H100:1 --partition Star --min-gpu-mem 70000"
    exit 2
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[FATAL] Model not found at $MODEL_PATH"
    exit 1
fi

# ============================================================
# Environment Setup
# ============================================================
echo "[Setup] Loading CUDA modules..."
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "[Warn] Could not load CUDA module"

# Conda setup — must use absolute paths; SLURM shells do NOT inherit login env
CONDA_BASE="/home/liujh/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate HiFloat4 2>/dev/null && echo "[Setup] Activated conda env: HiFloat4" || {
        conda activate base && echo "[Setup] Activated conda base (HiFloat4 not found)"
    }
elif [ -d "/home/liujh/HiFloat4/venv" ]; then
    source /home/liujh/HiFloat4/venv/bin/activate
    echo "[Setup] Activated venv"
else
    echo "[FATAL] No Python environment found"
    exit 1
fi

echo "[Setup] Checking PyTorch..."
python -c "import torch; print('PyTorch:', torch.__version__, '  CUDA:', torch.cuda.is_available())"

# Build HiF4 CUDA kernels
echo "[Setup] Building HiF4 CUDA kernels..."
bash hif4_gpu/build.sh 2>&1 | tail -5
echo ""

# ============================================================
# Step 1 — Quantize Wan 2.2
# ============================================================
echo "============================================"
echo "Step 1: Quantizing Wan 2.2 model"
echo "============================================"

QUANT_ARGS=(
    --model-path "$MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --quant-type "$QUANT_TYPE"
    --max-high-precision-layers "$MAX_HIGH_PRECISION"
    --rotation-mode "$ROTATION_MODE"
    --rotation-seed "$ROTATION_SEED"
    --device cuda
    --dtype bfloat16
)

$SKIP_SENSITIVITY  && QUANT_ARGS+=(--skip-sensitivity)
$NO_ROTATION       && QUANT_ARGS+=(--no-rotation)
$LOAD_ONLY         && QUANT_ARGS+=(--load-only)

echo "  Args: ${QUANT_ARGS[*]}"
python quantize_wan.py "${QUANT_ARGS[@]}"
echo "[Step 1] Done."
echo ""

if [ "$LOAD_ONLY" = "true" ]; then
    echo "[Info] Load-only mode enabled, stopping after Step 1 validation"
    echo "Finished at: $(date)"
    exit 0
fi

# ============================================================
# Step 2 — VBench Evaluation
# ============================================================
echo "============================================"
echo "Step 2: VBench evaluation"
echo "============================================"

EVAL_ARGS=(
    --model-path "$OUTPUT_DIR"
    --dataset "$DATASET_PATH"
    --output-dir "${OUTPUT_DIR}/vbench_results"
    --device cuda --dtype bfloat16 --seed 42
)

echo "  Args: ${EVAL_ARGS[*]}"
python run_quantized_wan.py "${EVAL_ARGS[@]}" || echo "[Warn] VBench exited non-zero"
echo "[Step 2] Done."
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================"
echo "Pipeline finished"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/" 2>/dev/null || true
echo ""
echo "Finished at: $(date)"