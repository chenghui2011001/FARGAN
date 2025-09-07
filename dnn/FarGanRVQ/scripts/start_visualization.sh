#!/usr/bin/env bash
set -euo pipefail

# FarGanRVQ 动态可视化启动脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 默认训练参数
DEFAULT_BATCH_SIZE=2048
DEFAULT_EPOCHS=400
DEFAULT_LR=5e-4
DEFAULT_SEQ_LEN=15

echo "🎨 FarGanRVQ 动态训练可视化工具"
echo "================================="
echo ""

# 检查环境
cd "${PROJECT_ROOT}"

echo "🔍 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python"
    exit 1
fi

echo "🔍 检查基础依赖..."
python -c "
try:
    import torch, numpy, matplotlib
    print('✅ 基础依赖检查通过')
except ImportError as e:
    print(f'❌ 缺少基础依赖: {e}')
    exit(1)
"

# 函数：获取训练参数
get_training_params() {
    echo ""
    echo "⚙️ 训练参数配置"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    read -p "📊 批量大小 (默认: $DEFAULT_BATCH_SIZE): " batch_size
    batch_size=${batch_size:-$DEFAULT_BATCH_SIZE}
    
    read -p "🔄 训练轮次 (默认: $DEFAULT_EPOCHS): " epochs
    epochs=${epochs:-$DEFAULT_EPOCHS}
    
    read -p "📈 学习率 (默认: $DEFAULT_LR): " lr
    lr=${lr:-$DEFAULT_LR}
    
    read -p "📏 序列长度 (默认: $DEFAULT_SEQ_LEN): " seq_len
    seq_len=${seq_len:-$DEFAULT_SEQ_LEN}
    
    echo ""
    echo "✅ 训练参数确认:"
    echo "  ├ 批量大小: $batch_size"
    echo "  ├ 训练轮次: $epochs"
    echo "  ├ 学习率: $lr"
    echo "  └ 序列长度: $seq_len"
    echo ""
    
    read -p "继续使用这些参数? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "❌ 已取消"
        exit 1
    fi
}

echo ""
echo "📊 可用的动态可视化方案:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  TensorBoard 监控 (推荐)"
echo "   ├ 🌟 专业级训练监控"
echo "   ├ 📊 实时损失曲线、学习率、梯度分布"
echo "   ├ 🖼️  模型结构图和权重可视化"
echo "   ├ 🔧 支持多实验对比"
echo "   └ 🌐 Web界面: http://localhost:6006"
echo ""
echo "2️⃣  Weights & Biases (云端)"
echo "   ├ ☁️  云端实时监控"
echo "   ├ 🎵 支持音频样本对比"
echo "   ├ 📱 移动端访问"
echo "   ├ 🤝 团队协作和实验管理"
echo "   └ 🔗 自动生成分享链接"
echo ""
echo "3️⃣  Streamlit Web界面"
echo "   ├ 🚀 快速启动，无需配置"
echo "   ├ 📈 实时图表和数据表格"
echo "   ├ 💾 数据导出功能"
echo "   ├ 📱 响应式设计"
echo "   └ 🌐 Web界面: http://localhost:8501"
echo ""
echo "4️⃣  静态分析报告"
echo "   ├ 📊 生成PNG图表和统计报告"
echo "   ├ 📋 详细的训练总结"
echo "   ├ 💽 离线查看"
echo "   └ 📄 支持导出多种格式"
echo ""
echo "5️⃣  终端实时监控"
echo "   ├ 🖥️  终端内实时显示"
echo "   ├ ⚡ 低资源占用"
echo "   ├ 🔢 纯文本输出"
echo "   └ 📊 ASCII图表"
echo ""

read -p "请选择可视化方案 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动 TensorBoard 监控..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 检查 tensorboard
        if ! python -c "import tensorboard" 2>/dev/null; then
            echo "⚠️  TensorBoard 未安装，正在安装..."
            pip install tensorboard
        fi
        
        echo "📋 启动选项:"
        echo "1) 查看现有日志"
        echo "2) 开始新的训练+监控"
        read -p "选择 (1-2): " tb_choice
        
        if [[ $tb_choice == "1" ]]; then
            LOG_DIR="dnn/FarGanRVQ/tensorboard_logs"
            echo "🌐 启动 TensorBoard..."
            echo "📊 访问: http://localhost:6006"
            tensorboard --logdir="${LOG_DIR}" --host=0.0.0.0 --port=6006
        else
            # 获取训练参数
            get_training_params
            
            echo "🏃 启动带 TensorBoard 的训练..."
            echo "📊 训练参数: 批量=${batch_size}, 轮次=${epochs}, 学习率=${lr}"
            
            python dnn/FarGanRVQ/training/train_with_tensorboard.py \
                --features data/out_features.f32 \
                --pcm data/out_speech.pcm \
                --batch-size "$batch_size" \
                --epochs "$epochs" \
                --lr "$lr" \
                --seq-len "$seq_len" \
                --log-interval 10 &
            
            TRAIN_PID=$!
            sleep 3
            
            echo "📊 启动 TensorBoard..."
            echo "🌐 访问: http://localhost:6006"
            echo "🔄 训练进程 PID: $TRAIN_PID"
            echo "💡 提示: 训练会在后台进行，TensorBoard会实时显示进度"
            tensorboard --logdir="dnn/FarGanRVQ/tensorboard_logs" --host=0.0.0.0 --port=6006
        fi
        ;;
        
    2)
        echo ""
        echo "☁️  启动 Weights & Biases 监控..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 检查 wandb
        if ! python -c "import wandb" 2>/dev/null; then
            echo "⚠️  Weights & Biases 未安装，正在安装..."
            pip install wandb
        fi
        
        # 检查是否已登录
        if ! wandb auth list >/dev/null 2>&1; then
            echo "🔑 请先登录 Weights & Biases:"
            echo "1. 访问 https://wandb.ai/authorize"
            echo "2. 复制 API key"
            wandb login
        fi
        
        # 获取训练参数
        get_training_params
        
        echo "🏃 启动带 WandB 的训练..."
        echo "📊 训练参数: 批量=${batch_size}, 轮次=${epochs}, 学习率=${lr}"
        
        python dnn/FarGanRVQ/training/train_with_wandb.py \
            --features data/out_features.f32 \
            --pcm data/out_speech.pcm \
            --batch-size "$batch_size" \
            --epochs "$epochs" \
            --lr "$lr" \
            --seq-len "$seq_len"
        ;;
        
    3)
        echo ""
        echo "🌐 启动 Streamlit Web 界面..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 检查依赖
        echo "🔍 检查 Streamlit 依赖..."
        python -c "
try:
    import streamlit, plotly, pandas
    print('✅ Streamlit 依赖检查通过')
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    print('📦 正在安装...')
    import subprocess
    subprocess.run(['pip', 'install', 'streamlit', 'plotly', 'pandas'])
"
        
        echo "🌐 启动 Streamlit 应用..."
        echo "📊 访问: http://localhost:8501"
        echo "🔄 应用会自动刷新监控数据"
        echo "💡 提示: 如果需要同时训练，请在另一个终端启动训练脚本"
        streamlit run dnn/FarGanRVQ/scripts/streamlit_monitor.py --server.port=8501 --server.address=0.0.0.0
        ;;
        
    4)
        echo ""
        echo "📊 生成静态分析报告..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python dnn/FarGanRVQ/scripts/analyze_training.py \
            --checkpoint-dir dnn/FarGanRVQ/checkpoints/test \
            --output-dir dnn/FarGanRVQ/analysis
        
        echo ""
        echo "✅ 报告生成完成!"
        echo "📁 查看结果: dnn/FarGanRVQ/analysis/"
        
        # 询问是否启动简单的HTTP服务器查看报告
        read -p "🌐 是否启动本地服务器查看报告? (y/N): " serve_report
        if [[ $serve_report =~ ^[Yy]$ ]]; then
            echo "🌐 启动本地服务器..."
            echo "📊 访问: http://localhost:8000/dnn/FarGanRVQ/analysis/"
            cd dnn/FarGanRVQ/analysis && python -m http.server 8000
        fi
        ;;
        
    5)
        echo ""
        echo "🖥️  启动终端实时监控..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 简单的终端监控
        CHECKPOINT_DIR="dnn/FarGanRVQ/checkpoints/test"
        
        echo "📊 实时监控 ${CHECKPOINT_DIR}"
        echo "⏹️  按 Ctrl+C 停止监控"
        echo ""
        
        # 添加训练启动选项
        read -p "🏃 是否同时启动训练? (y/N): " start_training
        if [[ $start_training =~ ^[Yy]$ ]]; then
            get_training_params
            
            echo "🚀 启动后台训练..."
            python dnn/FarGanRVQ/training/train_e2e.py \
                --features data/out_features.f32 \
                --pcm data/out_speech.pcm \
                --batch-size "$batch_size" \
                --epochs "$epochs" \
                --lr "$lr" \
                --seq-len "$seq_len" \
                --outdir "dnn/FarGanRVQ/checkpoints/test" > training.log 2>&1 &
            
            TRAIN_PID=$!
            echo "🔄 训练进程 PID: $TRAIN_PID"
            echo "📝 训练日志: training.log"
            echo ""
        fi
        
        while true; do
            clear
            echo "🎵 FarGanRVQ 训练监控 - $(date '+%Y-%m-%d %H:%M:%S')"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            if [[ -v TRAIN_PID ]]; then
                if kill -0 "$TRAIN_PID" 2>/dev/null; then
                    echo "🟢 后台训练进程: 运行中 (PID: $TRAIN_PID)"
                else
                    echo "🔴 后台训练进程: 已结束"
                fi
                echo ""
            fi
            
            if [[ -d "$CHECKPOINT_DIR" ]]; then
                count=$(find "$CHECKPOINT_DIR" -name "optfargan_*.pt" 2>/dev/null | wc -l)
                
                if [[ $count -gt 0 ]]; then
                    latest=$(find "$CHECKPOINT_DIR" -name "optfargan_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
                    latest_step=$(basename "$latest" | sed 's/optfargan_//; s/.pt//')
                    latest_time=$(stat -c %Y "$latest" 2>/dev/null)
                    current_time=$(date +%s)
                    time_diff=$((current_time - latest_time))
                    
                    # 状态判断
                    if [[ $time_diff -lt 60 ]]; then
                        status="🟢 训练中"
                    elif [[ $time_diff -lt 300 ]]; then
                        status="🟡 暂停中"
                    else
                        status="🔴 已停止"
                    fi
                    
                    echo "📊 状态: $status"
                    echo "📈 检查点数量: $count"
                    echo "⏰ 最新步数: $latest_step"
                    echo "🕒 最后更新: $time_diff 秒前"
                    
                    # 计算训练速度
                    if [[ $count -gt 5 ]]; then
                        files=($(find "$CHECKPOINT_DIR" -name "optfargan_*.pt" -printf '%T@ %p\n' | sort -n | tail -5))
                        first_time=$(echo "${files[0]}" | cut -d' ' -f1)
                        last_time=$(echo "${files[-1]}" | cut -d' ' -f1)
                        first_step=$(basename "$(echo "${files[0]}" | cut -d' ' -f2)" | sed 's/optfargan_//; s/.pt//')
                        last_step=$(basename "$(echo "${files[-1]}" | cut -d' ' -f2)" | sed 's/optfargan_//; s/.pt//')
                        
                        if [[ $(echo "$last_time - $first_time > 0" | bc 2>/dev/null || echo 0) == 1 ]]; then
                            speed=$(echo "scale=1; ($last_step - $first_step) * 60 / ($last_time - $first_time)" | bc 2>/dev/null || echo "0")
                            echo "⚡ 训练速度: ${speed} 步/分钟"
                        fi
                    fi
                    
                    # 显示最近几个检查点
                    echo ""
                    echo "📋 最近检查点:"
                    find "$CHECKPOINT_DIR" -name "optfargan_*.pt" -printf '%T+ %p\n' 2>/dev/null | sort | tail -5 | while read timestamp file; do
                        step=$(basename "$file" | sed 's/optfargan_//; s/.pt//')
                        size=$(stat -c%s "$file" 2>/dev/null | awk '{print int($1/1024/1024)"MB"}')
                        echo "  📁 步数 $step - $size - ${timestamp:0:19}"
                    done
                    
                    # ASCII 进度条
                    if [[ $count -gt 1 ]]; then
                        echo ""
                        echo "📊 训练进度 (基于检查点数量):"
                        # 根据设定的epochs估算进度
                        if [[ -v epochs ]]; then
                            estimated_total=$((epochs * 100))  # 估算每轮100个检查点
                            progress=$((count * 100 / estimated_total))
                        else
                            progress=$((count * 100 / 1000))  # 默认1000个检查点
                        fi
                        
                        progress=$((progress > 100 ? 100 : progress))  # 限制最大100%
                        filled=$((progress / 5))
                        printf "  ["
                        for ((i=1; i<=20; i++)); do
                            if [[ $i -le $filled ]]; then
                                printf "█"
                            else
                                printf "░"
                            fi
                        done
                        printf "] %d%%\n" $progress
                    fi
                    
                    # 显示最新日志
                    if [[ -f "training.log" ]]; then
                        echo ""
                        echo "📝 最新训练日志:"
                        tail -3 training.log 2>/dev/null | sed 's/^/  /'
                    fi
                else
                    echo "⏳ 等待检查点文件..."
                fi
            else
                echo "❌ 检查点目录不存在: $CHECKPOINT_DIR"
            fi
            
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🔄 刷新间隔: 5秒 | 按 Ctrl+C 退出监控"
            
            sleep 5
        done
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "👋 可视化监控已结束" 