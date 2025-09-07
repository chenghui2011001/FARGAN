#!/usr/bin/env python3
"""
FarGanRVQ 训练分析脚本
生成训练总结报告和静态图表
"""

import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 设置绘图参数
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [12, 8]

def scan_and_analyze_checkpoints(checkpoint_dir):
    """扫描并分析所有检查点文件"""
    pattern = os.path.join(checkpoint_dir, "optfargan_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print("❌ 未找到检查点文件")
        return None
    
    print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
    
    # 收集数据
    data = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        try:
            step = int(filename.split('_')[1].split('.')[0])
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mtime = os.path.getmtime(file_path)
            
            # 加载检查点信息
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                loss = checkpoint.get('loss', 0)
                epoch = checkpoint.get('epoch', 0)
            except:
                loss = 0
                epoch = 0
            
            data.append({
                'step': step,
                'loss': loss,
                'epoch': epoch,
                'size_mb': size_mb,
                'timestamp': mtime,
                'file': file_path
            })
        except (ValueError, IndexError) as e:
            print(f"⚠️ 跳过文件 {filename}: {e}")
            continue
    
    # 按步数排序
    data.sort(key=lambda x: x['step'])
    return data

def generate_training_report(data, output_dir):
    """生成训练报告"""
    if not data:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据序列
    steps = [d['step'] for d in data]
    losses = [d['loss'] for d in data]
    epochs = [d['epoch'] for d in data]
    timestamps = [d['timestamp'] for d in data]
    sizes = [d['size_mb'] for d in data]
    
    # 创建综合图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FarGanRVQ 训练分析报告', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('L1 损失')
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(steps) > 10:
        z = np.polyfit(steps, losses, 1)
        p = np.poly1d(z)
        ax1.plot(steps, p(steps), 'r--', alpha=0.8, label=f'趋势线 (斜率: {z[0]:.2e})')
        ax1.legend()
    
    # 2. 训练速度分析
    ax2 = axes[0, 1]
    if len(steps) > 1:
        # 计算训练速度 (步数/分钟)
        speeds = []
        speed_steps = []
        for i in range(1, len(steps)):
            dt = timestamps[i] - timestamps[i-1]
            ds = steps[i] - steps[i-1]
            if dt > 0:
                speed = (ds / dt) * 60  # 步数/分钟
                speeds.append(speed)
                speed_steps.append(steps[i])
        
        ax2.plot(speed_steps, speeds, 'g-', linewidth=2, alpha=0.8)
        ax2.set_title('训练速度')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('步数/分钟')
        ax2.grid(True, alpha=0.3)
        
        # 平均速度
        avg_speed = np.mean(speeds) if speeds else 0
        ax2.axhline(y=avg_speed, color='orange', linestyle='--', 
                   label=f'平均速度: {avg_speed:.1f} 步/分')
        ax2.legend()
    
    # 3. 模型大小变化
    ax3 = axes[1, 0]
    ax3.plot(steps, sizes, 'r-', linewidth=2, alpha=0.8)
    ax3.set_title('检查点文件大小')
    ax3.set_xlabel('训练步数')
    ax3.set_ylabel('文件大小 (MB)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 损失分布直方图
    ax4 = axes[1, 1]
    ax4.hist(losses, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('损失值分布')
    ax4.set_xlabel('L1 损失')
    ax4.set_ylabel('频次')
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax4.axvline(np.mean(losses), color='red', linestyle='--', 
               label=f'均值: {np.mean(losses):.6f}')
    ax4.axvline(np.median(losses), color='orange', linestyle='--', 
               label=f'中位数: {np.median(losses):.6f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 训练分析图表已保存: {plot_file}")
    
    # 生成详细统计报告
    generate_statistics_report(data, output_dir)
    
    plt.show()

def generate_statistics_report(data, output_dir):
    """生成详细统计报告"""
    if not data:
        return
    
    steps = [d['step'] for d in data]
    losses = [d['loss'] for d in data]
    timestamps = [d['timestamp'] for d in data]
    
    # 计算统计指标
    stats = {
        'training_summary': {
            'total_steps': max(steps) if steps else 0,
            'total_checkpoints': len(data),
            'start_time': datetime.fromtimestamp(min(timestamps)).isoformat() if timestamps else None,
            'end_time': datetime.fromtimestamp(max(timestamps)).isoformat() if timestamps else None,
            'total_duration_hours': (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) > 1 else 0
        },
        'loss_statistics': {
            'min_loss': float(np.min(losses)) if losses else 0,
            'max_loss': float(np.max(losses)) if losses else 0,
            'mean_loss': float(np.mean(losses)) if losses else 0,
            'median_loss': float(np.median(losses)) if losses else 0,
            'std_loss': float(np.std(losses)) if losses else 0,
            'final_loss': float(losses[-1]) if losses else 0
        },
        'performance_metrics': {
            'avg_steps_per_minute': 0,
            'estimated_completion_time': None,
            'model_size_mb': float(data[-1]['size_mb']) if data else 0
        }
    }
    
    # 计算训练速度
    if len(steps) > 1:
        total_time = max(timestamps) - min(timestamps)
        total_steps = max(steps) - min(steps)
        if total_time > 0:
            stats['performance_metrics']['avg_steps_per_minute'] = (total_steps / total_time) * 60
    
    # 保存统计报告
    stats_file = os.path.join(output_dir, 'training_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 生成文本报告
    report_file = os.path.join(output_dir, 'training_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FarGanRVQ 训练分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 训练概览:\n")
        f.write(f"  ├ 总步数: {stats['training_summary']['total_steps']:,}\n")
        f.write(f"  ├ 检查点数量: {stats['training_summary']['total_checkpoints']}\n")
        f.write(f"  ├ 训练时长: {stats['training_summary']['total_duration_hours']:.2f} 小时\n")
        f.write(f"  └ 平均速度: {stats['performance_metrics']['avg_steps_per_minute']:.1f} 步/分钟\n\n")
        
        f.write("📈 损失统计:\n")
        f.write(f"  ├ 最终损失: {stats['loss_statistics']['final_loss']:.6f}\n")
        f.write(f"  ├ 最小损失: {stats['loss_statistics']['min_loss']:.6f}\n")
        f.write(f"  ├ 平均损失: {stats['loss_statistics']['mean_loss']:.6f}\n")
        f.write(f"  ├ 标准差: {stats['loss_statistics']['std_loss']:.6f}\n")
        f.write(f"  └ 损失改善: {((stats['loss_statistics']['max_loss'] - stats['loss_statistics']['final_loss']) / stats['loss_statistics']['max_loss'] * 100) if stats['loss_statistics']['max_loss'] > 0 else 0:.2f}%\n\n")
        
        f.write("💾 模型信息:\n")
        f.write(f"  └ 模型大小: {stats['performance_metrics']['model_size_mb']:.1f} MB\n\n")
        
        f.write("🕒 时间信息:\n")
        f.write(f"  ├ 开始时间: {stats['training_summary']['start_time']}\n")
        f.write(f"  └ 结束时间: {stats['training_summary']['end_time']}\n")
    
    print(f"📋 统计报告已保存:")
    print(f"  ├ JSON: {stats_file}")
    print(f"  └ TXT: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ 训练分析工具')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='dnn/FarGanRVQ/checkpoints/test',
                       help='检查点文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='dnn/FarGanRVQ/analysis',
                       help='输出分析结果目录')
    
    args = parser.parse_args()
    
    print("🔍 开始分析训练数据...")
    
    # 扫描检查点
    data = scan_and_analyze_checkpoints(args.checkpoint_dir)
    
    if data:
        print(f"✅ 成功分析 {len(data)} 个检查点")
        
        # 生成报告
        generate_training_report(data, args.output_dir)
        
        print(f"\n📊 分析完成！结果保存在: {args.output_dir}")
    else:
        print("❌ 无法分析训练数据")

if __name__ == '__main__':
    main() 