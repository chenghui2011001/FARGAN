#!/usr/bin/env python3
"""
FarGanRVQ 训练过程可视化脚本
实时监控训练进度，绘制损失曲线和模型状态
"""

import argparse
import os
import sys
import time
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class TrainingVisualizer:
    def __init__(self, checkpoint_dir, refresh_interval=5):
        self.checkpoint_dir = checkpoint_dir
        self.refresh_interval = refresh_interval
        self.steps = []
        self.losses = []
        self.epochs = []
        self.timestamps = []
        
        # 创建图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('FarGanRVQ 训练监控面板', fontsize=16, fontweight='bold')
        
        # 损失曲线
        self.ax_loss = self.axes[0, 0]
        self.ax_loss.set_title('训练损失曲线')
        self.ax_loss.set_xlabel('训练步数')
        self.ax_loss.set_ylabel('L1 损失')
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='L1 Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # 训练速度
        self.ax_speed = self.axes[0, 1]
        self.ax_speed.set_title('训练速度')
        self.ax_speed.set_xlabel('时间')
        self.ax_speed.set_ylabel('步数/分钟')
        self.line_speed, = self.ax_speed.plot([], [], 'g-', linewidth=2, label='训练速度')
        self.ax_speed.legend()
        self.ax_speed.grid(True, alpha=0.3)
        
        # 检查点大小统计
        self.ax_size = self.axes[1, 0]
        self.ax_size.set_title('检查点文件大小')
        self.ax_size.set_xlabel('训练步数')
        self.ax_size.set_ylabel('文件大小 (MB)')
        self.line_size, = self.ax_size.plot([], [], 'r-', linewidth=2, label='模型大小')
        self.ax_size.legend()
        self.ax_size.grid(True, alpha=0.3)
        
        # 状态信息
        self.ax_info = self.axes[1, 1]
        self.ax_info.set_title('训练状态信息')
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.1, 0.9, '', transform=self.ax_info.transAxes, 
                                          fontsize=12, verticalalignment='top',
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
    def scan_checkpoints(self):
        """扫描检查点文件"""
        pattern = os.path.join(self.checkpoint_dir, "optfargan_*.pt")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return []
            
        # 解析步数并排序
        checkpoint_data = []
        for file_path in checkpoint_files:
            filename = os.path.basename(file_path)
            try:
                step = int(filename.split('_')[1].split('.')[0])
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                mtime = os.path.getmtime(file_path)
                checkpoint_data.append({
                    'step': step,
                    'file': file_path,
                    'size_mb': size_mb,
                    'timestamp': mtime
                })
            except (ValueError, IndexError):
                continue
                
        return sorted(checkpoint_data, key=lambda x: x['step'])
    
    def load_checkpoint_info(self, checkpoint_file):
        """加载检查点信息"""
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            return {
                'loss': checkpoint.get('loss', 0),
                'epoch': checkpoint.get('epoch', 0),
                'step': checkpoint.get('step', 0)
            }
        except Exception as e:
            print(f"⚠️ 无法加载检查点 {checkpoint_file}: {e}")
            return None
    
    def update_data(self):
        """更新训练数据"""
        checkpoints = self.scan_checkpoints()
        
        if not checkpoints:
            return False
            
        # 获取最新的几个检查点信息
        new_steps = []
        new_losses = []
        new_epochs = []
        new_timestamps = []
        new_sizes = []
        
        for ckpt_data in checkpoints[-50:]:  # 只看最近50个检查点
            info = self.load_checkpoint_info(ckpt_data['file'])
            if info:
                new_steps.append(info['step'])
                new_losses.append(info['loss'])
                new_epochs.append(info['epoch'])
                new_timestamps.append(ckpt_data['timestamp'])
                new_sizes.append(ckpt_data['size_mb'])
        
        self.steps = new_steps
        self.losses = new_losses
        self.epochs = new_epochs
        self.timestamps = new_timestamps
        self.sizes = new_sizes
        
        return len(self.steps) > 0
    
    def calculate_training_speed(self):
        """计算训练速度"""
        if len(self.steps) < 2 or len(self.timestamps) < 2:
            return []
            
        speeds = []
        times = []
        
        for i in range(1, len(self.steps)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            ds = self.steps[i] - self.steps[i-1]
            
            if dt > 0:
                speed = (ds / dt) * 60  # 步数/分钟
                speeds.append(speed)
                times.append(datetime.fromtimestamp(self.timestamps[i]))
        
        return times, speeds
    
    def update_plots(self):
        """更新所有图表"""
        if not self.steps:
            return
            
        # 更新损失曲线
        self.line_loss.set_data(self.steps, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # 更新训练速度
        times, speeds = self.calculate_training_speed()
        if speeds:
            self.line_speed.set_data(range(len(speeds)), speeds)
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()
        
        # 更新文件大小
        if hasattr(self, 'sizes'):
            self.line_size.set_data(self.steps, self.sizes)
            self.ax_size.relim()
            self.ax_size.autoscale_view()
        
        # 更新状态信息
        if self.steps:
            current_step = self.steps[-1]
            current_loss = self.losses[-1]
            current_epoch = self.epochs[-1] if self.epochs else 0
            
            # 计算平均速度
            avg_speed = np.mean(speeds[-10:]) if speeds else 0
            
            # ETA 估计
            if len(self.steps) >= 2:
                total_time = self.timestamps[-1] - self.timestamps[0]
                steps_done = self.steps[-1] - self.steps[0]
                if steps_done > 0:
                    time_per_step = total_time / steps_done
                    eta_total_steps = 10000  # 假设总目标步数
                    eta_remaining = (eta_total_steps - current_step) * time_per_step
                    eta_str = f"{eta_remaining/3600:.1f} 小时" if eta_remaining > 3600 else f"{eta_remaining/60:.1f} 分钟"
                else:
                    eta_str = "计算中..."
            else:
                eta_str = "计算中..."
            
            info_str = f"""
📊 训练状态总览
━━━━━━━━━━━━━━━━━━━━━━━

🎯 当前进度:
  └ 步数: {current_step:,}
  └ 轮次: {current_epoch}
  └ 损失: {current_loss:.6f}

⚡ 训练性能:
  └ 当前速度: {avg_speed:.1f} 步/分钟
  └ 预计完成: {eta_str}

💾 模型状态:
  └ 检查点数量: {len(self.steps)}
  └ 模型大小: {self.sizes[-1]:.1f} MB

🕒 最后更新: {datetime.now().strftime('%H:%M:%S')}
            """
            
            self.info_text.set_text(info_str.strip())
        
        plt.draw()
    
    def run_monitoring(self):
        """运行监控循环"""
        print(f"🔍 开始监控训练进度...")
        print(f"📁 检查点目录: {self.checkpoint_dir}")
        print(f"⏱️  刷新间隔: {self.refresh_interval} 秒")
        print(f"📊 可视化界面已启动，按 Ctrl+C 退出")
        
        try:
            while True:
                if self.update_data():
                    self.update_plots()
                    plt.pause(0.1)
                else:
                    print("⏳ 等待检查点文件...")
                
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
            plt.show()  # 保持窗口开启


def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ 训练可视化监控')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='dnn/FarGanRVQ/checkpoints/test',
                       help='检查点文件目录')
    parser.add_argument('--refresh', type=int, default=5,
                       help='刷新间隔（秒）')
    parser.add_argument('--save-plots', action='store_true',
                       help='保存图表到文件')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ 检查点目录不存在: {args.checkpoint_dir}")
        print("💡 请先运行训练脚本生成检查点文件")
        return
    
    # 创建可视化器
    visualizer = TrainingVisualizer(args.checkpoint_dir, args.refresh)
    
    # 如果需要保存图表
    if args.save_plots:
        output_dir = os.path.join(args.checkpoint_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        print(f"📊 图表将保存到: {output_dir}")
    
    # 运行监控
    visualizer.run_monitoring()


if __name__ == '__main__':
    main() 