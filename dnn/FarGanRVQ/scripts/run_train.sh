#!/usr/bin/env bash
set -euo pipefail

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/dnn/torch/fargan:${PYTHONPATH:-}"

echo "🚀 Starting FarGanRVQ training..."
echo "📁 Project root: ${PROJECT_ROOT}"
echo "🐍 Python path: ${PYTHONPATH}"

# 进入项目根目录
cd "${PROJECT_ROOT}"

# 检查环境
echo "🔍 Checking environment..."
python -c "
import sys
print('Python executable:', sys.executable)
print('Python path:')
for p in sys.path:
    print('  ', p)
"

# 检查关键文件是否存在
if [[ ! -f "dnn/torch/fargan/fargan.py" ]]; then
    echo "❌ fargan.py not found at dnn/torch/fargan/fargan.py"
    exit 1
fi

if [[ ! -f "data/out_features.f32" ]]; then
    echo "❌ Feature file not found at data/out_features.f32"
    echo "💡 Please check your data path"
    exit 1
fi

# 运行训练
echo "🎯 Starting training..."
python dnn/FarGanRVQ/training/train_e2e.py \
  --features "${PROJECT_ROOT}/data/out_features.f32" \
  --pcm "${PROJECT_ROOT}/data/out_speech.pcm" \
  --batch-size 8 \
  --epochs 1 \
  --lr 5e-4 \
  --outdir "${PROJECT_ROOT}/dnn/FarGanRVQ/checkpoints/basic" \
  "$@" 