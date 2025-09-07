#!/usr/bin/env bash
set -euo pipefail

# FarGanRVQ 快速训练启动脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "🚀 FarGanRVQ 快速训练启动器"
echo "================================"
echo ""

cd "${PROJECT_ROOT}"

echo "🎯 预设训练配置:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  快速测试 (小数据量)"
echo "   ├ 批量大小: 32"
echo "   ├ 训练轮次: 5"
echo "   ├ 学习率: 1e-3"
echo "   └ 用时: ~5分钟"
echo ""
echo "2️⃣  标准训练 (推荐)"
echo "   ├ 批量大小: 512"
echo "   ├ 训练轮次: 100"
echo "   ├ 学习率: 5e-4"
echo "   └ 用时: ~1-2小时"
echo ""
echo "3️⃣  高性能训练 (大批量)"
echo "   ├ 批量大小: 2048"
echo "   ├ 训练轮次: 400"
echo "   ├ 学习率: 5e-4"
echo "   └ 用时: ~4-6小时"
echo ""
echo "4️⃣  超长训练 (最佳质量)"
echo "   ├ 批量大小: 1024"
echo "   ├ 训练轮次: 1000"
echo "   ├ 学习率: 3e-4"
echo "   └ 用时: ~8-12小时"
echo ""
echo "5️⃣  自定义参数"
echo "   └ 手动输入所有参数"
echo ""

read -p "请选择训练配置 (1-5): " config_choice

case $config_choice in
    1)
        BATCH_SIZE=32
        EPOCHS=5
        LR=1e-3
        SEQ_LEN=15
        CONFIG_NAME="快速测试"
        ;;
    2)
        BATCH_SIZE=512
        EPOCHS=100
        LR=5e-4
        SEQ_LEN=15
        CONFIG_NAME="标准训练"
        ;;
    3)
        BATCH_SIZE=2048
        EPOCHS=400
        LR=5e-4
        SEQ_LEN=15
        CONFIG_NAME="高性能训练"
        ;;
    4)
        BATCH_SIZE=1024
        EPOCHS=1000
        LR=3e-4
        SEQ_LEN=15
        CONFIG_NAME="超长训练"
        ;;
    5)
        CONFIG_NAME="自定义配置"
        echo ""
        echo "⚙️ 自定义训练参数"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        read -p "📊 批量大小 (建议: 32-4096): " BATCH_SIZE
        read -p "🔄 训练轮次 (建议: 10-1000): " EPOCHS
        read -p "📈 学习率 (建议: 1e-4 到 1e-3): " LR
        read -p "📏 序列长度 (建议: 10-20): " SEQ_LEN
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
echo "🔄 训练轮次: ${EPOCHS}"
echo "📈 学习率: ${LR}"
echo "📏 序列长度: ${SEQ_LEN}"
echo ""

# 检查数据文件
FEATURES_FILE="data/out_features.f32"
PCM_FILE="data/out_speech.pcm"

if [[ ! -f "$FEATURES_FILE" ]]; then
    echo "❌ 特征文件不存在: $FEATURES_FILE"
    echo "💡 请确保已准备好训练数据"
    exit 1
fi

if [[ ! -f "$PCM_FILE" ]]; then
    echo "❌ 音频文件不存在: $PCM_FILE"
    echo "💡 请确保已准备好训练数据"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

# 选择训练模式
echo "🎯 训练模式选择:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) 基础训练 (只保存检查点)"
echo "2) TensorBoard 训练 (实时可视化)"
echo "3) Weights & Biases 训练 (云端监控)"
echo ""

read -p "选择训练模式 (1-3): " train_mode

# 生成输出目录名
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="dnn/FarGanRVQ/checkpoints/${CONFIG_NAME// /_}_${TIMESTAMP}"

echo ""
echo "📁 输出目录: ${OUTPUT_DIR}"
echo ""

case $train_mode in
    1)
        echo "🏃 启动基础训练..."
        python dnn/FarGanRVQ/training/train_e2e.py \
            --features "$FEATURES_FILE" \
            --pcm "$PCM_FILE" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --seq-len "$SEQ_LEN" \
            --outdir "$OUTPUT_DIR" \
            --num-workers 8 \
            --prefetch-factor 4 \
            --drop-last \
            --grad-accum-steps 2 \
            --enable-tf32 \
            --mixed-precision \
            --compile \
            --log-timing
        ;;
        
    2)
        echo "🏃 启动 TensorBoard 训练..."
        
        # 检查 TensorBoard
        if ! python -c "import tensorboard" 2>/dev/null; then
            echo "📦 安装 TensorBoard..."
            pip install tensorboard
        fi
        
        # 启动训练（后台）
        LOG_DIR="dnn/FarGanRVQ/tensorboard_logs/${CONFIG_NAME// /_}_${TIMESTAMP}"
        
        python dnn/FarGanRVQ/training/train_with_tensorboard.py \
            --features "$FEATURES_FILE" \
            --pcm "$PCM_FILE" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --seq-len "$SEQ_LEN" \
            --log-dir "$LOG_DIR" \
            --num-workers 8 \
            --prefetch-factor 4 \
            --drop-last \
            --grad-accum-steps 2 \
            --enable-tf32 \
            --mixed-precision \
            --compile \
            --log-timing &
        
        TRAIN_PID=$!
        
        echo "🔄 训练进程 PID: $TRAIN_PID"
        echo "📊 TensorBoard 日志: $LOG_DIR"
        
        sleep 3
        
        echo "🌐 启动 TensorBoard..."
        echo "📊 访问: http://localhost:6006"
        echo "💡 提示: 按 Ctrl+C 停止 TensorBoard（训练会继续在后台运行）"
        
        tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006
        ;;
        
    3)
        echo "🏃 启动 Weights & Biases 训练..."
        
        # 检查 WandB
        if ! python -c "import wandb" 2>/dev/null; then
            echo "📦 安装 Weights & Biases..."
            pip install wandb
        fi
        
        # 检查登录
        if ! wandb auth list >/dev/null 2>&1; then
            echo "🔑 请先登录 Weights & Biases:"
            wandb login
        fi
        
        python dnn/FarGanRVQ/training/train_with_wandb.py \
            --features "$FEATURES_FILE" \
            --pcm "$PCM_FILE" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --seq-len "$SEQ_LEN" \
            --num-workers 8 \
            --prefetch-factor 4 \
            --drop-last \
            --grad-accum-steps 2 \
            --enable-tf32 \
            --mixed-precision \
            --log-timing
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 训练任务完成!"
echo "📁 结果保存在: ${OUTPUT_DIR}"

# 训练完成后的操作
echo ""
echo "📊 后续操作:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) 生成训练报告: python dnn/FarGanRVQ/scripts/analyze_training.py --checkpoint-dir ${OUTPUT_DIR}"
echo "2) 启动 Streamlit 监控: streamlit run dnn/FarGanRVQ/scripts/streamlit_monitor.py"
echo "3) 评估模型: python dnn/FarGanRVQ/evaluation/eval_pipeline.py" 