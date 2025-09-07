#!/usr/bin/env python3
"""
FarGanRVQ 训练脚本 - 集成 TensorBoard 动态可视化 + GPU支持
支持实时监控损失、模型结构、训练速度等，GPU加速训练
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
fargan_path = os.path.join(project_root, 'dnn/torch/fargan')
sys.path.insert(0, fargan_path)

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.optimized_fargan import OptimizedFarGan


def setup_device():
    """设置训练设备并显示相关信息"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 使用GPU训练: {gpu_name}")
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        print(f"🗑️ GPU缓存已清理")
        
        # 启用优化选项
        torch.backends.cudnn.benchmark = True
        print(f"⚡ cuDNN benchmark已启用")
        
        return device, gpu_name, gpu_memory
    else:
        device = torch.device('cpu')
        print(f"⚠️ 未检测到GPU，使用CPU训练")
        print(f"💡 提示: GPU训练会显著提升速度")
        return device, "CPU", 0


def collate_fn(batch):
    """高效数据加载"""
    features_list = []
    data_list = []
    
    for features, periods, data, lpc in batch:
        features_list.append(features)
        data_list.append(data)
    
    features = torch.from_numpy(np.array(features_list)).float()
    data = torch.from_numpy(np.array(data_list)).float()
    
    return features, data


class TensorBoardTrainer:
    def __init__(self, model, optimizer, loss_fn, writer, args, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writer = writer
        self.args = args
        self.device = device
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.losses = []
        self.learning_rates = []
        
    def log_model_info(self, sample_input):
        """记录模型结构和参数"""
        # 模型结构图
        try:
            self.writer.add_graph(self.model, sample_input)
            print("✅ 模型结构图已添加到 TensorBoard")
        except Exception as e:
            print(f"⚠️ 无法添加模型图: {e}")
        
        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.writer.add_text('Model/Info', f"""
        **模型参数统计:**
        - 总参数: {total_params:,}
        - 可训练参数: {trainable_params:,}
        - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)
        - 训练设备: {self.device}
        """)
        
        print(f"📊 模型参数: {trainable_params:,} (可训练) / {total_params:,} (总计)")
    
    def log_training_step(self, loss, batch_idx, data_size):
        """记录单步训练信息"""
        self.step += 1
        current_loss = loss.item()
        self.losses.append(current_loss)
        
        # 基础损失曲线
        self.writer.add_scalar('Training/Loss', current_loss, self.step)
        
        # 学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        self.writer.add_scalar('Training/Learning_Rate', current_lr, self.step)
        
        # 训练速度
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.step / elapsed_time if elapsed_time > 0 else 0
        self.writer.add_scalar('Performance/Steps_Per_Second', steps_per_sec, self.step)
        
        # 移动平均损失
        if len(self.losses) >= 10:
            avg_loss_10 = np.mean(self.losses[-10:])
            self.writer.add_scalar('Training/Loss_MA10', avg_loss_10, self.step)
        
        if len(self.losses) >= 100:
            avg_loss_100 = np.mean(self.losses[-100:])
            self.writer.add_scalar('Training/Loss_MA100', avg_loss_100, self.step)
        
        # 批次大小和数据吞吐量
        samples_per_sec = (data_size * steps_per_sec) if steps_per_sec > 0 else 0
        self.writer.add_scalar('Performance/Samples_Per_Second', samples_per_sec, self.step)
        
        # GPU/内存使用
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            
            self.writer.add_scalar('System/GPU_Memory_Used_GB', memory_used, self.step)
            self.writer.add_scalar('System/GPU_Memory_Cached_GB', memory_cached, self.step)
            self.writer.add_scalar('System/GPU_Utilization', gpu_util, self.step)
        else:
            import psutil
            memory = psutil.virtual_memory()
            self.writer.add_scalar('System/RAM_Used_GB', memory.used / 1024**3, self.step)
            self.writer.add_scalar('System/RAM_Percent', memory.percent, self.step)
    
    def log_epoch_summary(self, epoch_loss, epoch_time):
        """记录每轮训练总结"""
        self.epoch += 1
        
        # 轮次损失
        self.writer.add_scalar('Training/Epoch_Loss', epoch_loss, self.epoch)
        self.writer.add_scalar('Performance/Epoch_Time_Minutes', epoch_time / 60, self.epoch)
        
        # 损失改善趋势
        if len(self.losses) >= 100:
            recent_loss = np.mean(self.losses[-50:])
            older_loss = np.mean(self.losses[-100:-50])
            improvement = ((older_loss - recent_loss) / older_loss * 100) if older_loss > 0 else 0
            self.writer.add_scalar('Training/Loss_Improvement_Percent', improvement, self.epoch)
        
        # 预计完成时间
        if self.epoch > 0:
            avg_epoch_time = (time.time() - self.start_time) / self.epoch
            remaining_epochs = self.args.epochs - self.epoch
            eta_hours = (remaining_epochs * avg_epoch_time) / 3600
            self.writer.add_scalar('Performance/ETA_Hours', eta_hours, self.epoch)
    
    def log_model_weights(self):
        """记录模型权重分布"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 权重分布
                self.writer.add_histogram(f'Weights/{name}', param.data, self.step)
                # 梯度分布
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, self.step)
                # 梯度范数
                grad_norm = param.grad.data.norm().item()
                self.writer.add_scalar(f'Gradients/{name}_norm', grad_norm, self.step)


def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ TensorBoard GPU训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件 .f32')
    parser.add_argument('--pcm', type=str, required=True, help='音频文件 .pcm')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seq-len', type=int, default=15)
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--log-interval', type=int, default=10, help='记录间隔步数')
    parser.add_argument('--weight-log-interval', type=int, default=100, help='权重记录间隔')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='训练设备')
    parser.add_argument('--mixed-precision', action='store_true', help='启用混合精度训练')
    parser.add_argument('--compile', action='store_true', help='启用模型编译优化')
    # 新增：性能/并行相关参数
    parser.add_argument('--num-workers', type=int, default=-1, help='DataLoader 工作线程数，-1 表示自动')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='每个 worker 预取批次数（num_workers>0 时生效）')
    parser.add_argument('--drop-last', action='store_true', help='丢弃最后一个不足批量的批次')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='梯度累积步数，用于放大有效批量')
    parser.add_argument('--enable-tf32', action='store_true', help='在支持的 GPU 上启用 TF32 加速')
    parser.add_argument('--log-timing', action='store_true', help='记录数据加载/计算耗时拆分')
    
    args = parser.parse_args()
    
    print(f"🎯 FarGanRVQ TensorBoard GPU训练启动")
    print(f"{'='*50}")
    
    # 设置设备
    if args.device == 'auto':
        device, gpu_name, gpu_memory = setup_device()
    else:
        device = torch.device(args.device)
        gpu_name = "Manual" if device.type == 'cuda' else "CPU"
        gpu_memory = 0
        print(f"🎯 手动指定设备: {device}")
    
    # 可选：启用 TF32 加速（Ampere+）
    if device.type == 'cuda' and args.enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("⚙️ 已启用 TF32 加速 (matmul/cudnn)")
        except Exception as e:
            print(f"⚠️ 启用 TF32 失败: {e}")
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    device_name = gpu_name.replace(' ', '_') if device.type == 'cuda' else 'CPU'
    log_dir = os.path.join(args.log_dir, f'run_{device_name}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"🌐 启动命令: tensorboard --logdir={args.log_dir}")
    
    # 数据集设置
    frame_size = 160
    seq_len = args.seq_len
    T_frames = seq_len * 2 + 4
    F_used = 20
    N_samples = frame_size * seq_len
    
    print(f"\n📋 训练配置:")
    print(f"  ├ 批量大小: {args.batch_size}")
    print(f"  ├ 序列长度: {seq_len} 帧")
    print(f"  ├ 学习率: {args.lr}")
    print(f"  ├ 输出样本数: {N_samples}")
    print(f"  ├ 训练设备: {device}")
    print(f"  ├ 混合精度: {args.mixed_precision}")
    print(f"  └ 模型编译: {args.compile}")
    
    # 数据加载器优化
    if args.num_workers < 0:
        num_workers = 4 if device.type == 'cuda' else 2
    else:
        num_workers = args.num_workers
    pin_memory = device.type == 'cuda'
    persistent_workers = num_workers > 0
    
    # 加载数据
    ds = FARGANDataset(
        feature_file=args.features,
        signal_file=args.pcm,
        frame_size=frame_size,
        sequence_length=seq_len,
        lookahead=1,
        nb_used_features=F_used,
        nb_features=36,
    )
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        drop_last=args.drop_last,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None
    )
    
    # 模型和优化器
    model = OptimizedFarGan(in_features=F_used, cond_dim=96)
    model = model.to(device)
    
    # 模型编译优化
    if args.compile and hasattr(torch, 'compile'):
        print(f"🔥 启用模型编译优化...")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    loss_fn = nn.L1Loss()
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    
    # 记录超参数和设备信息
    hparams = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seq_len': seq_len,
        'model_params': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'mixed_precision': args.mixed_precision,
        'model_compile': args.compile,
        'grad_accum_steps': args.grad_accum_steps,
        'num_workers': num_workers,
        'prefetch_factor': args.prefetch_factor,
        'drop_last': args.drop_last,
        'enable_tf32': args.enable_tf32,
        'log_timing': args.log_timing,
    }
    
    if device.type == 'cuda':
        hparams.update({
            'gpu_name': gpu_name,
            'gpu_memory_gb': gpu_memory
        })
    
    writer.add_hparams(hparams, {})
    
    # 创建训练器
    trainer = TensorBoardTrainer(model, optimizer, loss_fn, writer, args, device)
    
    # 记录模型信息（使用第一个批次）
    sample_batch = next(iter(dl))
    sample_features = sample_batch[0][:1].to(device)  # 只取一个样本并移动到设备
    trainer.log_model_info(sample_features[:, :T_frames, :])
    
    print(f"\n🚀 开始训练...")
    
    model.train()
    
    last_end_time = time.time()
    optimizer_step = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, data) in enumerate(dl):
            # 统计数据加载耗时
            now_time = time.time()
            data_time = now_time - last_end_time
            
            # 移动数据到设备
            features = features.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)
            
            # 数据预处理
            y = data[:, :N_samples] if data.shape[1] >= N_samples else data
            
            # 前向传播 + 反向传播（支持梯度累积）
            if args.mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    y_hat = model(features[:, :T_frames, :], target_length=N_samples)
                    loss = loss_fn(y_hat, y)
                    loss_to_backward = loss / max(1, args.grad_accum_steps)
                
                if (batch_idx % args.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_to_backward).backward()
                
                take_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
                if take_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step += 1
            else:
                y_hat = model(features[:, :T_frames, :], target_length=N_samples)
                loss = loss_fn(y_hat, y)
                loss_to_backward = loss / max(1, args.grad_accum_steps)
                
                if (batch_idx % args.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                loss_to_backward.backward()
                
                take_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
                if take_step:
                    optimizer.step()
                    optimizer_step += 1
            
            # 记录训练步骤
            trainer.log_training_step(loss, batch_idx, y.shape[0])
            
            # 可选：记录权重
            if trainer.step % args.weight_log_interval == 0:
                trainer.log_model_weights()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 统计计算耗时
            end_time = time.time()
            compute_time = end_time - now_time
            last_end_time = end_time
            if args.log_timing:
                writer.add_scalar('Performance/Data_Load_Time_s', data_time, trainer.step)
                writer.add_scalar('Performance/Compute_Time_s', compute_time, trainer.step)
            
            # 打印进度
            if trainer.step % args.log_interval == 0:
                elapsed_time = time.time() - trainer.start_time
                steps_per_sec = trainer.step / elapsed_time
                
                if device.type == 'cuda':
                    mem_info = f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB"
                else:
                    import psutil
                    mem_info = f"RAM: {psutil.virtual_memory().percent:.1f}%"
                
                extra_timing = ''
                if args.log_timing:
                    extra_timing = f" | data {data_time*1000:.1f}ms, compute {compute_time*1000:.1f}ms"
                
                print(f'Epoch {epoch+1}/{args.epochs} Step {trainer.step} | '
                      f'Loss: {loss.item():.6f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                      f'Speed: {steps_per_sec:.2f} steps/s | {mem_info}{extra_timing}')
        
        # 若最后不足累积步，也需执行一次 step（已在逻辑中覆盖）
        
        # 记录轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        trainer.log_epoch_summary(avg_epoch_loss, epoch_time)
        
        print(f'✅ Epoch {epoch+1} 完成 | 平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s')
    
    # 记录最终模型状态
    total_time = time.time() - trainer.start_time
    writer.add_text('Training/Completion', f"""
    **训练完成!**
    - 总步数: {trainer.step:,}
    - 总时长: {total_time/3600:.2f} 小时
    - 最终损失: {trainer.losses[-1]:.6f}
    - 最佳损失: {min(trainer.losses):.6f}
    - 训练设备: {device}
    - 平均速度: {trainer.step/total_time:.2f} steps/s
    """)
    
    writer.close()
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🗑️ GPU缓存已清理")
    
    print(f"\n🎉 训练完成!")
    print(f"📊 查看结果: tensorboard --logdir={args.log_dir}")
    print(f"⚡ 平均速度: {trainer.step/total_time:.2f} steps/s")


if __name__ == '__main__':
    main() 