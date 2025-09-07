#!/usr/bin/env bash
set -euo pipefail

# FarGanRVQ 高 GPU 利用率训练启动脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "🚀 FarGanRVQ 高 GPU 利用率训练"
echo "=================================="
echo ""

cd "${PROJECT_ROOT}"

# 检查 GPU
if ! nvidia-smi >/dev/null 2>&1; then
    echo "❌ 未检测到 NVIDIA GPU 或驱动"
    exit 1
fi

echo "💾 GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# 数据文件检查
FEATURES_FILE="data/out_features.f32"
PCM_FILE="data/out_speech.pcm"

if [[ ! -f "$FEATURES_FILE" ]]; then
    echo "❌ 特征文件不存在: $FEATURES_FILE"
    exit 1
fi

if [[ ! -f "$PCM_FILE" ]]; then
    echo "❌ 音频文件不存在: $PCM_FILE"
    exit 1
fi

echo "📋 GPU 利用率优化选项:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  标准优化 (推荐首次尝试)"
echo "   ├ 批量大小: 4096"
echo "   ├ 大模型: 300万+ 参数"
echo "   ├ 数据线程: 12"
echo "   └ 梯度累积: 8 步"
echo ""
echo "2️⃣  激进优化 (最大GPU利用率)"
echo "   ├ 批量大小: 8192"  
echo "   ├ 大模型: 300万+ 参数"
echo "   ├ 数据线程: 16"
echo "   └ 梯度累积: 16 步"
echo ""
echo "3️⃣  保守优化 (稳定训练)"
echo "   ├ 批量大小: 2048"
echo "   ├ 大模型: 300万+ 参数"
echo "   ├ 数据线程: 8"
echo "   └ 梯度累积: 4 步"
echo ""
echo "4️⃣  自定义参数"
echo ""

read -p "请选择配置 (1-4): " config_choice

case $config_choice in
    1)
        BATCH_SIZE=4096
        MODEL_SIZE="large"
        NUM_WORKERS=12
        GRAD_ACCUM=8
        CONFIG_NAME="标准优化"
        ;;
    2)
        BATCH_SIZE=8192
        MODEL_SIZE="large"
        NUM_WORKERS=16
        GRAD_ACCUM=16
        CONFIG_NAME="激进优化"
        ;;
    3)
        BATCH_SIZE=2048
        MODEL_SIZE="large"
        NUM_WORKERS=8
        GRAD_ACCUM=4
        CONFIG_NAME="保守优化"
        ;;
    4)
        CONFIG_NAME="自定义配置"
        echo ""
        echo "⚙️ 自定义GPU优化参数"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        read -p "📊 批量大小 (建议: 2048-8192): " BATCH_SIZE
        echo "模型大小选择:"
        echo "  small: 65万参数 (低GPU使用)"
        echo "  large: 300万+参数 (高GPU使用)"
        read -p "🏗️ 模型大小 (small/large): " MODEL_SIZE
        read -p "🔄 数据线程数 (建议: 8-16): " NUM_WORKERS
        read -p "📈 梯度累积步数 (建议: 4-16): " GRAD_ACCUM
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 选择的配置: ${CONFIG_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 批量大小: ${BATCH_SIZE}"
echo "🏗️ 模型大小: ${MODEL_SIZE}"
echo "🔄 数据线程: ${NUM_WORKERS}"
echo "📈 梯度累积: ${GRAD_ACCUM}"
echo ""

# GPU 优化环境变量
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "🚀 启动高GPU利用率训练..."
echo "💡 建议同时运行 nvidia-smi dmon 监控GPU使用率"
echo ""

# 启动训练
python dnn/FarGanRVQ/training/train_high_gpu.py \
    --features "$FEATURES_FILE" \
    --pcm "$PCM_FILE" \
    --batch-size "$BATCH_SIZE" \
    --model-size "$MODEL_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --epochs 100 \
    --seq-len 20 \
    --lr 1e-3

echo ""
echo "🎉 训练完成!"
echo ""
echo "📊 查看GPU利用率建议:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) 实时监控: nvidia-smi dmon"
echo "2) 详细统计: nvidia-smi pmon -i 0 -s um"
echo "3) TensorBoard: tensorboard --logdir=dnn/FarGanRVQ/tensorboard_logs" 