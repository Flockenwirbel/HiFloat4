SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/quant_cy/base/cusrc"
python setup.py build_ext --inplace 
cd "$SCRIPT_DIR"
