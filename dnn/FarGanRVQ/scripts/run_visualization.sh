#!/usr/bin/env bash
set -euo pipefail

# FarGanRVQ 训练可视化启动脚本

# 设置项目路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "🎨 FarGanRVQ 训练可视化工具"
echo "================================"
echo "📁 项目根目录: ${PROJECT_ROOT}"

# 检查环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python"
    exit 1
fi

# 检查必要的包
echo "🔍 检查依赖包..."
python -c "
try:
    import torch, matplotlib, numpy
    print('✅ 核心依赖包检查通过')
except ImportError as e:
    print(f'❌ 缺少依赖包: {e}')
    exit(1)
"

# 默认检查点目录
CHECKPOINT_DIR="${PROJECT_ROOT}/dnn/FarGanRVQ/checkpoints/test"

echo ""
echo "请选择可视化模式:"
echo "1) 实时监控 (动态更新图表)"
echo "2) 静态分析 (生成报告和图表)"
echo "3) 查看现有检查点数量"
echo ""

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "🔄 启动实时训练监控..."
        cd "${PROJECT_ROOT}"
        python dnn/FarGanRVQ/scripts/visualize_training.py \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --refresh 5
        ;;
    2)
        echo "📊 生成训练分析报告..."
        cd "${PROJECT_ROOT}"
        python dnn/FarGanRVQ/scripts/analyze_training.py \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --output-dir "dnn/FarGanRVQ/analysis"
        ;;
    3)
        echo "📋 检查点文件统计:"
        if [[ -d "${CHECKPOINT_DIR}" ]]; then
            count=$(find "${CHECKPOINT_DIR}" -name "optfargan_*.pt" | wc -l)
            echo "  ├ 检查点数量: $count"
            
            if [[ $count -gt 0 ]]; then
                latest=$(find "${CHECKPOINT_DIR}" -name "optfargan_*.pt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
                latest_step=$(basename "$latest" | sed 's/optfargan_//; s/.pt//')
                latest_time=$(stat -c %y "$latest" | cut -d'.' -f1)
                echo "  ├ 最新步数: $latest_step"
                echo "  └ 最后更新: $latest_time"
                
                # 计算总大小
                total_size=$(find "${CHECKPOINT_DIR}" -name "optfargan_*.pt" -exec stat -c%s {} + | awk '{s+=$1} END {print s/1024/1024}')
                echo "  └ 总大小: ${total_size%.0f} MB"
            fi
        else
            echo "  └ 检查点目录不存在"
        fi
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 可视化完成!" 