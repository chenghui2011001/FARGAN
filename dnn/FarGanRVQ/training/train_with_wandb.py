#!/usr/bin/env python3
"""
FarGanRVQ 训练脚本 - 集成 Weights & Biases 动态可视化
支持云端实时监控、实验管理、模型版本控制
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ Weights & Biases 可用")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Weights & Biases 未安装，运行: pip install wandb")


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


class WandBTrainer:
    def __init__(self, model, optimizer, loss_fn, args, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args
        self.device = device
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.losses = []
        
        if WANDB_AVAILABLE:
            # 获取设备信息
            device_info = {"device": str(device)}
            if device.type == 'cuda':
                device_info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
                })
            
            # 初始化 wandb
            wandb.init(
                project="fargan-rvq",
                name=f"optfargan_{datetime.now().strftime('%m%d_%H%M')}_{device.type}",
                config={
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "seq_len": args.seq_len,
                    "model": "OptimizedFarGan",
                    "optimizer": "AdamW",
                    "loss": "L1Loss",
                    "mixed_precision": args.mixed_precision,
                    "model_compile": args.compile,
                    "grad_accum_steps": args.grad_accum_steps,
                    "num_workers": num_workers,
                    "prefetch_factor": args.prefetch_factor,
                    "drop_last": args.drop_last,
                    "enable_tf32": args.enable_tf32,
                    "log_timing": args.log_timing,
                    **device_info
                }
            )
            
            # 记录模型架构
            wandb.watch(self.model, log="all", log_freq=100)
            print("🌐 Weights & Biases 已初始化")
            print(f"📊 实验链接: {wandb.run.url}")
    
    def log_training_step(self, loss, batch_idx, data_size):
        """记录单步训练信息"""
        self.step += 1
        current_loss = loss.item()
        self.losses.append(current_loss)
        
        if not WANDB_AVAILABLE:
            return
        
        # 基础指标
        metrics = {
            "train/loss": current_loss,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
            "train/step": self.step,
            "train/epoch": self.epoch
        }
        
        # 训练速度
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            metrics["performance/steps_per_second"] = self.step / elapsed_time
            metrics["performance/samples_per_second"] = (data_size * self.step) / elapsed_time
        
        # 移动平均损失
        if len(self.losses) >= 10:
            metrics["train/loss_ma10"] = np.mean(self.losses[-10:])
        if len(self.losses) >= 100:
            metrics["train/loss_ma100"] = np.mean(self.losses[-100:])
        
        # GPU/内存使用
        if self.device.type == 'cuda':
            metrics["system/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
            metrics["system/gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            if hasattr(torch.cuda, 'utilization'):
                metrics["system/gpu_utilization"] = torch.cuda.utilization()
        else:
            import psutil
            memory = psutil.virtual_memory()
            metrics["system/ram_used_gb"] = memory.used / 1024**3
            metrics["system/ram_percent"] = memory.percent
        
        wandb.log(metrics, step=self.step)
    
    def log_epoch_summary(self, epoch_loss, epoch_time):
        """记录每轮训练总结"""
        self.epoch += 1
        
        if not WANDB_AVAILABLE:
            return
        
        metrics = {
            "epoch/loss": epoch_loss,
            "epoch/time_minutes": epoch_time / 60,
            "epoch/number": self.epoch
        }
        
        # 损失改善趋势
        if len(self.losses) >= 100:
            recent_loss = np.mean(self.losses[-50:])
            older_loss = np.mean(self.losses[-100:-50])
            if older_loss > 0:
                improvement = ((older_loss - recent_loss) / older_loss * 100)
                metrics["epoch/loss_improvement_percent"] = improvement
        
        # ETA 估计
        if self.epoch > 0:
            avg_epoch_time = (time.time() - self.start_time) / self.epoch
            remaining_epochs = self.args.epochs - self.epoch
            eta_hours = (remaining_epochs * avg_epoch_time) / 3600
            metrics["epoch/eta_hours"] = eta_hours
        
        wandb.log(metrics, step=self.step)
    
    def log_audio_samples(self, original, generated, sample_rate=16000):
        """记录音频样本对比"""
        if not WANDB_AVAILABLE:
            return
        
        try:
            # 只记录第一个样本进行对比
            orig_audio = original[0].detach().cpu().numpy()
            gen_audio = generated[0].detach().cpu().numpy()
            
            wandb.log({
                "audio/original": wandb.Audio(orig_audio, sample_rate=sample_rate),
                "audio/generated": wandb.Audio(gen_audio, sample_rate=sample_rate),
            }, step=self.step)
        except Exception as e:
            print(f"⚠️ 音频记录失败: {e}")
    
    def log_loss_distribution(self):
        """记录损失分布"""
        if not WANDB_AVAILABLE or len(self.losses) < 10:
            return
        
        wandb.log({
            "charts/loss_histogram": wandb.Histogram(self.losses[-100:])
        }, step=self.step)
    
    def save_model_artifact(self, checkpoint_path):
        """保存模型为 wandb artifact"""
        if not WANDB_AVAILABLE:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"fargan-model",
                type="model",
                description=f"FarGan model at step {self.step}",
                metadata={
                    "step": self.step,
                    "epoch": self.epoch,
                    "loss": self.losses[-1] if self.losses else 0
                }
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"💾 模型已保存为 wandb artifact")
        except Exception as e:
            print(f"⚠️ Artifact 保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ Weights & Biases 训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件 .f32')
    parser.add_argument('--pcm', type=str, required=True, help='音频文件 .pcm')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seq-len', type=int, default=15)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--audio-log-interval', type=int, default=500, help='音频记录间隔')
    parser.add_argument('--wandb-project', type=str, default='fargan-rvq')
    parser.add_argument('--wandb-entity', type=str, help='wandb 团队/用户名')
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
    
    if not WANDB_AVAILABLE:
        print("❌ 请先安装 wandb: pip install wandb")
        return
    
    print(f"🎯 FarGanRVQ WandB GPU训练启动")
    print(f"{'='*50}")
    
    # 设置设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 使用GPU训练: {gpu_name}")
            print(f"💾 GPU内存: {gpu_memory:.1f} GB")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            print(f"⚠️ 未检测到GPU，使用CPU训练")
    else:
        device = torch.device(args.device)
        print(f"🎯 手动指定设备: {device}")
    
    # 可选：启用 TF32 加速（Ampere+）
    if device.type == 'cuda' and args.enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("⚙️ 已启用 TF32 加速 (matmul/cudnn)")
        except Exception as e:
            print(f"⚠️ 启用 TF32 失败: {e}")
    
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
    
    # 创建训练器
    trainer = WandBTrainer(model, optimizer, loss_fn, args, device)
    
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
            
            # 定期记录音频样本
            if trainer.step % args.audio_log_interval == 0:
                trainer.log_audio_samples(y, y_hat)
            
            # 定期记录损失分布
            if trainer.step % 200 == 0:
                trainer.log_loss_distribution()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 统计计算耗时
            end_time = time.time()
            compute_time = end_time - now_time
            last_end_time = end_time
            if args.log_timing and WANDB_AVAILABLE:
                wandb.log({
                    'Performance/Data_Load_Time_s': data_time,
                    'Performance/Compute_Time_s': compute_time
                }, step=trainer.step)
            
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
        
        # 记录轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        trainer.log_epoch_summary(avg_epoch_loss, epoch_time)
        
        print(f'✅ Epoch {epoch+1} 完成 | 平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s')
        
        # 保存检查点和 artifact
        if epoch % 5 == 0:  # 每5轮保存一次
            checkpoint_path = f"dnn/FarGanRVQ/checkpoints/wandb_epoch_{epoch}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'step': trainer.step,
                'device': str(device),
                'mixed_precision': args.mixed_precision
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            trainer.save_model_artifact(checkpoint_path)
    
    # 训练完成
    total_time = time.time() - trainer.start_time
    print(f"\n🎉 训练完成!")
    print(f"📊 总耗时: {total_time/3600:.2f} 小时")
    print(f"⚡ 平均速度: {trainer.step/total_time:.2f} steps/s")
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🗑️ GPU缓存已清理")
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main() 