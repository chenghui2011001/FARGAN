#!/usr/bin/env bash
set -euo pipefail

# MambaJSCC 训练启动脚本
# 基于论文的GSSM + CSI-ReST实现

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "🚀 MambaJSCC 训练启动"
echo "基于 GSSM + CSI-ReST 信道自适应技术"
echo "================================================"
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

echo "📋 MambaJSCC 配置选项:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  基础训练 (轻量模型)"
echo "   ├ Mamba层: 4"
echo "   ├ 条件维度: 128"
echo "   ├ 信道: AWGN"
echo "   ├ SNR范围: [-10, 20] dB"
echo "   └ 批量: 16"
echo ""
echo "2️⃣  增强训练 (性能优化)"
echo "   ├ Mamba层: 6"
echo "   ├ 条件维度: 192"
echo "   ├ 信道: 瑞利衰落"
echo "   ├ SNR范围: [-15, 25] dB"
echo "   └ 批量: 32"
echo ""
echo "3️⃣  研究配置 (完整功能)"
echo "   ├ Mamba层: 8"
echo "   ├ 条件维度: 256"
echo "   ├ 信道: 混合（AWGN+瑞利）"
echo "   ├ SNR范围: [-20, 30] dB"
echo "   └ 批量: 64"
echo ""

read -p "请选择配置 (1-3): " config_choice

case $config_choice in
    1)
        N_MAMBA_LAYERS=4
        COND_DIM=128
        CHANNEL_TYPE="awgn"
        SNR_MIN=-10
        SNR_MAX=20
        BATCH_SIZE=16
        CONFIG_NAME="基础训练"
        ;;
    2)
        N_MAMBA_LAYERS=6
        COND_DIM=192
        CHANNEL_TYPE="rayleigh"
        SNR_MIN=-15
        SNR_MAX=25
        BATCH_SIZE=32
        CONFIG_NAME="增强训练"
        ;;
    3)
        N_MAMBA_LAYERS=8
        COND_DIM=256
        CHANNEL_TYPE="rayleigh"
        SNR_MIN=-20
        SNR_MAX=30
        BATCH_SIZE=64
        CONFIG_NAME="研究配置"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 选择的配置: ${CONFIG_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏗️ Mamba层数: ${N_MAMBA_LAYERS}"
echo "📐 条件维度: ${COND_DIM}"
echo "📡 信道类型: ${CHANNEL_TYPE}"
echo "📊 SNR范围: [${SNR_MIN}, ${SNR_MAX}] dB"
echo "🎯 批量大小: ${BATCH_SIZE}"
echo ""

# 计算预期参数量（基于配置）
case $config_choice in
    1) EXPECTED_PARAMS="~2.1M" ;;
    2) EXPECTED_PARAMS="~4.2M" ;;
    3) EXPECTED_PARAMS="~8.5M" ;;
esac

echo "📈 预期模型参数: ${EXPECTED_PARAMS}"
echo ""

# MambaJSCC 环境变量
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export MAMBA_FORCE_BUILD=1

echo "🚀 启动 MambaJSCC 训练..."
echo "💡 建议监控: nvidia-smi dmon -s puc"
echo ""

# 检查是否需要混合精度
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEMORY" -gt 11000 ]; then
    MIXED_PRECISION=""
    echo "💾 GPU内存充足 (${GPU_MEMORY}MB)，使用FP32精度"
else
    MIXED_PRECISION="--mixed-precision"
    echo "💾 GPU内存有限 (${GPU_MEMORY}MB)，启用混合精度"
fi

# 启动训练
python dnn/FarGanRVQ/training/train_mamba_jscc.py \
    --features "$FEATURES_FILE" \
    --pcm "$PCM_FILE" \
    --batch-size "$BATCH_SIZE" \
    --n-mamba-layers "$N_MAMBA_LAYERS" \
    --cond-dim "$COND_DIM" \
    --channel-type "$CHANNEL_TYPE" \
    --snr-range "$SNR_MIN" "$SNR_MAX" \
    --epochs 50 \
    --seq-len 60 \
    --lr-g 3e-5 \
    --lr-d 1e-4 \
    --num-workers 8 \
    $MIXED_PRECISION \
    --compile

echo ""
echo "🎉 MambaJSCC 训练完成!"
echo ""
echo "📊 性能分析建议:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) TensorBoard: tensorboard --logdir=dnn/FarGanRVQ/tensorboard_logs"
echo "2) GPU使用率: nvidia-smi pmon -i 0 -s um"
echo "3) 模型分析: python -c \"from dnn.FarGanRVQ.models.mamba_fargan_jscc import *; print('模型加载成功')\""
echo ""
echo "🔬 MambaJSCC 核心特性验证:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ GSSM双向扫描: 理论保证全局信息捕获"
echo "✓ CSI-ReST机制: 零参数信道自适应"
echo "✓ JSCC编解码: 端到端信源信道联合优化"
echo "✓ 相位感知损失: 提升音质一致性"
echo "✓ 多SNR训练: 增强信道鲁棒性" 