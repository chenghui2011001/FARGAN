#!/bin/bash

echo "🚀 MambaJSCC 高性能训练启动器"
echo "=================================="
echo ""

# 检查CUDA可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ 错误: 未检测到NVIDIA GPU"
    exit 1
fi

echo "📊 GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

echo "选择高性能训练模式:"
echo "1) 🔥 极致性能 (批量256, 编译+TF32)"
echo "2) ⚡ 高性能 (批量128, 编译)"  
echo "3) 🎯 平衡性能 (批量64, 优化设置)"
echo "4) 🔧 自定义配置"
echo ""

read -p "请选择模式 [1-4]: " choice

case $choice in
    1)
        echo "🔥 启动极致性能模式..."
        BATCH_SIZE=256
        GRAD_ACCUM=8
        NUM_WORKERS=8
        PREFETCH=8
        COMPILE="--compile"
        TF32="--enable-tf32"
        CACHE="--enable-cache"
        ;;
    2)
        echo "⚡ 启动高性能模式..."
        BATCH_SIZE=128
        GRAD_ACCUM=4
        NUM_WORKERS=6
        PREFETCH=6
        COMPILE="--compile"
        TF32=""
        CACHE="--enable-cache"
        ;;
    3)
        echo "🎯 启动平衡性能模式..."
        BATCH_SIZE=64
        GRAD_ACCUM=2
        NUM_WORKERS=4
        PREFETCH=4
        COMPILE=""
        TF32=""
        CACHE=""
        ;;
    4)
        echo "🔧 自定义配置模式..."
        read -p "批量大小 [64]: " BATCH_SIZE
        BATCH_SIZE=${BATCH_SIZE:-64}
        read -p "梯度累积步数 [2]: " GRAD_ACCUM
        GRAD_ACCUM=${GRAD_ACCUM:-2}
        read -p "数据加载workers [4]: " NUM_WORKERS
        NUM_WORKERS=${NUM_WORKERS:-4}
        read -p "启用编译优化? [y/N]: " ENABLE_COMPILE
        COMPILE=""
        if [[ $ENABLE_COMPILE =~ ^[Yy]$ ]]; then
            COMPILE="--compile"
        fi
        PREFETCH=4
        TF32=""
        CACHE=""
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "📋 训练配置:"
echo "├ 批量大小: $BATCH_SIZE"
echo "├ 梯度累积: $GRAD_ACCUM 步"
echo "├ Workers: $NUM_WORKERS"
echo "├ 预取因子: $PREFETCH"
echo "├ 编译优化: $([ -n "$COMPILE" ] && echo "启用" || echo "禁用")"
echo "├ TF32加速: $([ -n "$TF32" ] && echo "启用" || echo "禁用")"
echo "└ 缓存优化: $([ -n "$CACHE" ] && echo "启用" || echo "禁用")"
echo ""

# 生成时间戳标识
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_NAME="mamba_fast_${TIMESTAMP}"

echo "🏃‍♂️ 正在启动高性能训练..."
echo "📊 TensorBoard: tensorboard --logdir=dnn/FarGanRVQ/tensorboard_logs/"
echo ""

# 启动训练
python dnn/FarGanRVQ/training/train_enhanced_mamba.py \
    --features data/out_features.f32 \
    --pcm data/out_speech.pcm \
    --output dnn/FarGanRVQ/checkpoints/ \
    --log-dir dnn/FarGanRVQ/tensorboard_logs/${LOG_NAME} \
    --enable-csi \
    --snr-range -15 25 \
    --channel-prob 0.7 \
    --n-mamba-layers 6 \
    --batch-size $BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM \
    --num-workers $NUM_WORKERS \
    --prefetch-factor $PREFETCH \
    --mixed-precision \
    --pin-memory \
    --drop-last \
    --persistent-workers \
    --lr-g 3e-5 \
    --lr-d 1e-4 \
    --epochs 100 \
    --log-interval 10 \
    --save-interval 1000 \
    --sequence-length 60 \
    $COMPILE \
    $TF32 \
    $CACHE

echo ""
echo "🎉 训练完成！" 