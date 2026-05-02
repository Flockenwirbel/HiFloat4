#!/bin/bash
#
# W4A4 Quantized Wan 2.2 Pipeline — One-Click Execution Script
#
# Run directly on fuse0 — auto-requests a GPU node via srun if needed.
#
#   cd ~/HiFloat4
#   bash run_pipeline.sh
#
# Full pipeline:
#   Step 1: Quantize Wan 2.2 Transformer (W4A4 + Hadamard rotation)
#   Step 2: Transformer proxy evaluation (cosine similarity, etc.)
#   Step 3: I2V video generation (first frame + caption -> new video)
#   Step 4: VBench evaluation (assess generated video quality)
#
# Options:
#   bash run_pipeline.sh --quant-type mxfp4 --no-rotation --skip-sensitivity
#   bash run_pipeline.sh --enable-rotation --enable-sensitivity
#   bash run_pipeline.sh --max-high-precision-layers 2 --num-videos 50
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

# --- Quantization parameters ---
QUANT_TYPE="hifx4"          # hifx4 (max 2 high-precision layers) or mxfp4 (max 5)
MAX_HIGH_PRECISION=2         # Number of sensitive layers to keep in BF16
ROTATION_MODE="none"         # Hadamard rotation mode: none, pad, or block
ROTATION_SEED=42             # Random seed for rotation (reproducibility)
SKIP_SENSITIVITY=true        # Skip sensitivity analysis (faster)
NO_ROTATION=true             # Disable Hadamard rotation (recommended for now)
LOAD_ONLY=false              # Only load and validate, skip evaluation
SKIP_QUANTIZE=false          # Skip Steps 1-2, go directly to video generation
EVAL_STEPS=4                 # Number of proxy evaluation steps
VIDEO_METRIC_SAMPLES=8       # Number of video metric samples

# --- I2V video generation parameters ---
GENERATE_VIDEOS=true         # Whether to generate I2V videos
NUM_VIDEOS=0                 # Number of videos to generate (0 = all 488 available)
NUM_FRAMES=81                # Frames per video (81 frames ~ 5s @ 16fps)
RESOLUTION="480p"            # Resolution: 480p (480x832) or 720p (720x1280)
GUIDANCE_SCALE=5.0           # CFG guidance strength (higher = more text-adherent)
NUM_INFERENCE_STEPS=50       # Diffusion denoising steps (50 for quality)
FPS=16                       # Output video frame rate

# --- VBench evaluation parameters ---
VBENCH_EVAL=true             # Whether to run VBench evaluation

# --- SLURM resource request ---
SRUN_GRES="gpu:H100:1"
SRUN_CPUS="8"
SRUN_MEM="64G"
SRUN_TIME="8:00:00"
SRUN_PARTITION="Star"
SRUN_EXTRA=""
MIN_GPU_MEM_MIB=70000

# Save original CLI args (needed for srun forwarding)
ORIGINAL_ARGS=("$@")

# ============================================================
# Parse CLI overrides
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)        MODEL_PATH="$2";        shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --quant-type)        QUANT_TYPE="$2";        shift 2 ;;
        --max-high-precision-layers) MAX_HIGH_PRECISION="$2"; shift 2 ;;
        --rotation-mode)     ROTATION_MODE="$2";     shift 2 ;;
        --rotation-seed)     ROTATION_SEED="$2";     shift 2 ;;
        --no-rotation)       NO_ROTATION=true;       shift   ;;
        --enable-rotation)   NO_ROTATION=false;      shift   ;;
        --skip-sensitivity)  SKIP_SENSITIVITY=true;  shift   ;;
        --enable-sensitivity) SKIP_SENSITIVITY=false; shift  ;;
        --eval-steps)        EVAL_STEPS="$2";        shift 2 ;;
        --video-metric-samples) VIDEO_METRIC_SAMPLES="$2"; shift 2 ;;
        --num-videos)        NUM_VIDEOS="$2";        shift 2 ;;
        --num-frames)        NUM_FRAMES="$2";        shift 2 ;;
        --resolution)        RESOLUTION="$2";        shift 2 ;;
        --guidance-scale)    GUIDANCE_SCALE="$2";    shift 2 ;;
        --inference-steps)   NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --fps)               FPS="$2";               shift 2 ;;
        --no-generate)       GENERATE_VIDEOS=false;  shift   ;;
        --no-vbench)         VBENCH_EVAL=false;      shift   ;;
        --gpu)               SRUN_GRES="gpu:$2";     shift 2 ;;
        --gres)              SRUN_GRES="$2";         shift 2 ;;
        --partition)         SRUN_PARTITION="$2";    shift 2 ;;
        --constraint)        SRUN_EXTRA="--constraint=$2"; shift 2 ;;
        --mem)               SRUN_MEM="$2";          shift 2 ;;
        --time)              SRUN_TIME="$2";         shift 2 ;;
        --min-gpu-mem)       MIN_GPU_MEM_MIB="$2";   shift 2 ;;
        --load-only)         LOAD_ONLY=true;         shift   ;;
        --skip-quantize)    SKIP_QUANTIZE=true;      shift   ;;
        *) echo "Unknown option: $1"; shift ;;
    esac
done

# Guard high-precision layer count by quant type constraint
if [ "$QUANT_TYPE" = "hifx4" ] && [ "$MAX_HIGH_PRECISION" -gt 2 ]; then
    echo "[Warn] HiF4 allows max 2 high-precision layers, capping to 2"
    MAX_HIGH_PRECISION=2
elif [ "$QUANT_TYPE" = "mxfp4" ] && [ "$MAX_HIGH_PRECISION" -gt 5 ]; then
    echo "[Warn] MXFP4 allows max 5 high-precision layers, capping to 5"
    MAX_HIGH_PRECISION=5
fi

# ============================================================
# Auto-detect GPU — relaunch via srun if needed
# ============================================================
GPU_AVAILABLE=false
if [ -n "${SLURM_JOB_ID:-}" ]; then
    if nvidia-smi -L &>/dev/null; then
        GPU_AVAILABLE=true
    else
        echo "[Warn] SLURM_JOB_ID=${SLURM_JOB_ID} but nvidia-smi -L failed"
    fi
else
    if nvidia-smi -L &>/dev/null; then
        GPU_AVAILABLE=true
    fi
fi

if [ "$GPU_AVAILABLE" != "true" ]; then
    echo ""
    echo "================================================================"
    echo "  No usable GPU detected -- submitting to SLURM"
    echo "================================================================"
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
    echo "[FATAL] GPU memory insufficient: ${GPU_MEM_TOTAL_MIB} MiB < ${MIN_GPU_MEM_MIB} MiB"
    echo "        Try a larger GPU, e.g.:"
    echo "        bash run_pipeline.sh --gres gpu:H100:1"
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

# Conda setup -- must use absolute paths; SLURM shells do NOT inherit login env
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

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Build HiF4 CUDA kernels
echo "[Setup] Building HiF4 CUDA kernels..."
bash hif4_gpu/build.sh 2>&1 | tail -5
echo ""

# ============================================================
# Step 1 -- Quantize Wan 2.2 Transformer (skip if --skip-quantize)
# ============================================================
if [ "$SKIP_QUANTIZE" = "true" ]; then
    echo "[Step 1] Skipped quantization (--skip-quantize)"
    echo ""
else
#
# Applies W4A4 quantization with Hadamard rotation to all linear
# layers, except the top-K most sensitive ones (kept in BF16).
# The Hadamard transform smooths outlier channels, reducing
# quantization error from cosine ~0.895 to ~0.912.
# ============================================================
echo "============================================"
echo "Step 1: Quantizing Wan 2.2 Transformer"
echo "============================================"
echo "  Quant type       : $QUANT_TYPE (W4A4)"
echo "  High-prec layers : $MAX_HIGH_PRECISION"
echo "  Rotation mode    : $ROTATION_MODE"
echo "  Rotation seed    : $ROTATION_SEED"
echo ""

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

if [ "$SKIP_SENSITIVITY" = "true" ]; then
    QUANT_ARGS+=(--skip-sensitivity)
else
    QUANT_ARGS+=(--enable-sensitivity)
fi

if [ "$NO_ROTATION" = "true" ]; then
    QUANT_ARGS+=(--no-rotation)
else
    QUANT_ARGS+=(--enable-rotation)
fi

$LOAD_ONLY && QUANT_ARGS+=(--load-only)

echo "  Command: python quantize_wan.py ${QUANT_ARGS[*]}"
python quantize_wan.py "${QUANT_ARGS[@]}"
echo "[Step 1] Quantization done."
echo ""
fi  # end SKIP_QUANTIZE guard

if [ "$LOAD_ONLY" = "true" ]; then
    echo "[Info] Load-only mode, stopping after Step 1 validation"
    echo "Finished at: $(date)"
    exit 0
fi

# ============================================================
# Step 2 -- Transformer Proxy Evaluation (skip if --skip-quantize)
# ============================================================
if [ "$SKIP_QUANTIZE" = "true" ]; then
    echo "[Step 2] Skipped proxy evaluation (--skip-quantize)"
    echo ""
else
#
# Runs random inputs through the quantized transformer and
# measures cosine similarity vs. the original BF16 output.
# This is a fast proxy metric (not a full video generation).
# ============================================================
echo "============================================"
echo "Step 2: Transformer proxy evaluation"
echo "============================================"

EVAL_ARGS=(
    --model-path "$OUTPUT_DIR"
    --dataset "$DATASET_PATH"
    --output-dir "${OUTPUT_DIR}/vbench_results"
    --device cuda --dtype bfloat16 --seed 42
    --eval-steps "$EVAL_STEPS"
    --video-metric-samples "$VIDEO_METRIC_SAMPLES"
)

echo "  Command: python run_quantized_wan.py ${EVAL_ARGS[*]}"
python run_quantized_wan.py "${EVAL_ARGS[@]}" || echo "[Warn] Proxy evaluation exited non-zero"
echo "[Step 2] Proxy evaluation done."
echo ""
fi  # end SKIP_QUANTIZE guard

# ============================================================
# Step 3 -- I2V Video Generation
#
# For each OpenS2V sample:
#   - Extract the first frame of the source video as the image input
#   - Use the caption as the text prompt
#   - Generate a new video using the quantized Wan 2.2 I2V pipeline
#
# Parameters explained:
#   NUM_VIDEOS (100)      - VBench recommends >=100 for stable scores
#   NUM_FRAMES (81)       - Standard Wan 2.2 frame count (~5s @ 16fps)
#   RESOLUTION (480p)     - 480x832, the default Wan 2.2 resolution
#   GUIDANCE_SCALE (5.0)  - Classifier-Free Guidance strength;
#                           higher values make output more faithful to
#                           the text prompt but reduce diversity
#   NUM_INFERENCE_STEPS (50) - Diffusion denoising steps;
#                              50 is the standard quality/speed tradeoff
#   FPS (16)              - Output video frame rate
# ============================================================
if [ "$GENERATE_VIDEOS" = "true" ]; then
    echo "============================================"
    echo "Step 3: I2V video generation"
    echo "============================================"
    echo "  Num videos       : $NUM_VIDEOS"
    echo "  Frames/video     : $NUM_FRAMES"
    echo "  Resolution       : $RESOLUTION"
    echo "  Guidance scale   : $GUIDANCE_SCALE"
    echo "  Inference steps  : $NUM_INFERENCE_STEPS"
    echo "  FPS              : $FPS"
    echo "  Output dir       : ${OUTPUT_DIR}/generated_videos"
    echo ""

    GEN_ARGS=(
        --model-path "$MODEL_PATH"
        --quantized-path "$OUTPUT_DIR"
        --dataset "$DATASET_PATH"
        --output-dir "${OUTPUT_DIR}/generated_videos"
        --device cuda --dtype bfloat16
        --num-videos "$NUM_VIDEOS"
        --num-frames "$NUM_FRAMES"
        --resolution "$RESOLUTION"
        --guidance-scale "$GUIDANCE_SCALE"
        --num-inference-steps "$NUM_INFERENCE_STEPS"
        --fps "$FPS"
        --seed 42
    )

    echo "  Command: python generate_videos.py ${GEN_ARGS[*]}"
    python generate_videos.py "${GEN_ARGS[@]}" || echo "[Warn] Video generation exited non-zero"
    echo "[Step 3] Video generation done."
    echo ""
else
    echo "[Step 3] Skipped video generation (--no-generate or GENERATE_VIDEOS=false)"
    echo ""
fi

# ============================================================
# Step 4 -- VBench Evaluation
#
# Evaluates the generated videos using VBench metrics.
# Measures quality dimensions like subject consistency, background
# consistency, motion smoothness, aesthetic quality, etc.
# ============================================================
if [ "$VBENCH_EVAL" = "true" ] && [ "$GENERATE_VIDEOS" = "true" ]; then
    echo "============================================"
    echo "Step 4: VBench evaluation"
    echo "============================================"

    VBENCH_DIR="${OUTPUT_DIR}/generated_videos"
    VBENCH_OUT="${OUTPUT_DIR}/vbench_results"

    echo "  Video dir  : $VBENCH_DIR"
    echo "  Output dir : $VBENCH_OUT"
    echo ""

    python -c "
import json, os, glob
from vbench import VBench

video_dir = '$VBENCH_DIR'
output_dir = '$VBENCH_OUT'
os.makedirs(output_dir, exist_ok=True)

# Check for generated videos
videos = glob.glob(os.path.join(video_dir, '*.mp4'))
print(f'[VBench] Found {len(videos)} videos')

if len(videos) == 0:
    print('[VBench] No videos to evaluate, skipping')
else:
    # Run VBench I2V evaluation
    vb = VBench(device='cuda')

    # Load generation metadata for captions
    meta_path = os.path.join(video_dir, 'generation_metadata.json')
    captions = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Run VBench evaluation
    result = vb.evaluate(
        video_path=video_dir,
        output_path=output_dir,
    )

    result_path = os.path.join(output_dir, 'vbench_i2v_results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[VBench] Results saved to {result_path}')
    print(f'[VBench] Total score: {result.get(\"total_score\", \"N/A\")}')
" 2>&1 || echo "[Warn] VBench evaluation failed (may need additional dependencies)"
    echo "[Step 4] VBench evaluation done."
    echo ""
else
    echo "[Step 4] Skipped VBench evaluation"
    echo ""
fi

# ============================================================
# Summary
# ============================================================
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "File listing:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || true
echo ""

if [ "$GENERATE_VIDEOS" = "true" ]; then
    echo "Generated videos:"
    ls "${OUTPUT_DIR}/generated_videos/"*.mp4 2>/dev/null | wc -l | xargs -I{} echo "  {} video files"
fi
echo ""
echo "Finished at: $(date)"