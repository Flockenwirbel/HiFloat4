#!/bin/bash
# Build HiF4 CUDA kernels on a GPU node (fuse2)
# Usage: sbatch build_on_gpu.sh  OR  srun --partition=fuse2 --gres=gpu:1 bash build_on_gpu.sh

set -euo pipefail

# --- Environment setup ---
source /home/liujh/miniconda3/etc/profile.d/conda.sh
conda activate HiFloat4

echo "=== Build Environment ==="
echo "Host: $(hostname)"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "Python: $(which python)"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'NOT FOUND (will look for it)')"

# Try to find nvcc
if ! command -v nvcc &>/dev/null; then
    # Common CUDA install paths
    for cuda_dir in /usr/local/cuda-12* /usr/local/cuda-11* /usr/local/cuda \
                    /opt/cuda /opt/conda/envs/*/bin; do
        if [ -x "$cuda_dir/bin/nvcc" ]; then
            export PATH="$cuda_dir/bin:$PATH"
            export CUDA_HOME="$cuda_dir"
            break
        fi
    done

    # Last resort: try nvcc from conda
    if ! command -v nvcc &>/dev/null; then
        pip install --quiet cuda-toolkit 2>/dev/null || true
        if [ -d "$CONDA_PREFIX/bin" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
            export PATH="$CONDA_PREFIX/bin:$PATH"
            export CUDA_HOME="$CONDA_PREFIX"
        fi
    fi
fi

echo "Using nvcc: $(which nvcc)"
nvcc --version | head -2

echo ""
echo "=== Building quant_cy CUDA extensions ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/hif4_gpu/quant_cy/base/cusrc"

# Clean previous build artifacts
if [ -d build ]; then
    rm -rf build
fi

# Build the extension
python setup.py build_ext --inplace

echo ""
echo "=== Verifying build ==="
cd "$SCRIPT_DIR"
python -c "
import sys
sys.path.insert(0, '.')
from hif4_gpu.quant_cy import QLinear, QType
print('quant_cy module built and imported successfully')
print('QType:', dir(QType))
"

echo ""
echo "=== Build complete ==="

# Track that build was done
touch "$SCRIPT_DIR/.build_success"