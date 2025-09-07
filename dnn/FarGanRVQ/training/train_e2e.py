#!/usr/bin/env python3
"""
FarGanRVQ 基础训练脚本 - 支持GPU加速
自动检测GPU并优化内存使用
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time

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
        
    else:
        device = torch.device('cpu')
        print(f"⚠️ 未检测到GPU，使用CPU训练")
        print(f"💡 提示: GPU训练会显著提升速度")
    
    return device


def get_memory_usage(device):
    """获取内存使用情况"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    else:
        import psutil
        memory = psutil.virtual_memory()
        used = memory.used / 1024**3
        total = memory.total / 1024**3
        return f"RAM: {used:.2f}GB / {total:.2f}GB ({memory.percent:.1f}%)"


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


def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ GPU训练 (OptimizedFarGan)')
    parser.add_argument('--features', type=str, required=True, help='feature_file .f32')
    parser.add_argument('--pcm', type=str, required=True, help='signal_file .pcm int16')
    parser.add_argument('--batch-size', type=int, default=2048, help='批量大小')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮次')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--outdir', type=str, default='dnn/FarGanRVQ/checkpoints', help='输出目录')
    parser.add_argument('--seq-len', type=int, default=15, help='序列长度（帧数）')
    parser.add_argument('--save-interval', type=int, default=10, help='保存检查点间隔步数')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='训练设备 (auto: 自动检测)')
    parser.add_argument('--mixed-precision', action='store_true', help='启用混合精度训练')
    parser.add_argument('--compile', action='store_true', help='启用模型编译优化 (PyTorch 2.0+)')
    # 新增：性能/并行相关参数
    parser.add_argument('--num-workers', type=int, default=-1, help='DataLoader 工作线程数，-1 表示自动')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='每个 worker 预取批次数（num_workers>0 时生效）')
    parser.add_argument('--drop-last', action='store_true', help='丢弃最后一个不足批量的批次')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='梯度累积步数，用于放大有效批量')
    parser.add_argument('--enable-tf32', action='store_true', help='在支持的 GPU 上启用 TF32 加速')
    parser.add_argument('--log-timing', action='store_true', help='记录数据加载/计算耗时拆分')
    args = parser.parse_args()

    print(f"🎯 FarGanRVQ GPU训练启动")
    print(f"{'='*50}")
    
    # 设置设备
    if args.device == 'auto':
        device = setup_device()
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
    print(f"  ├ 混合精度: {args.mixed_precision}")
    print(f"  ├ 模型编译: {args.compile}")
    print(f"  ├ 梯度累积: {args.grad_accum_steps} 步")
    print(f"  └ 训练设备: {device}")
    
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

    # 模型参数计算
    T_frames = seq_len * 2 + 4  # FarGan 数据集的特征帧数计算
    F_used = 20  # 使用的特征维度
    N_samples = frame_size * seq_len  # 期望的输出样本数
    
    print(f"\n🔧 Model configuration:")
    print(f"   Input features: [B, {T_frames}, {F_used}]")
    print(f"   Expected output: [B, {N_samples}]")
    
    # 创建模型并移动到设备
    model = OptimizedFarGan(in_features=F_used, cond_dim=96)
    model = model.to(device)
    
    # 模型编译优化 (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print(f"🔥 启用模型编译优化...")
        model = torch.compile(model)
    
    # 优化器和损失函数
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    loss_fn = nn.L1Loss()
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: {trainable_params:,} (可训练) / {total_params:,} (总计)")
    print(f"💾 模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    model.train()
    step = 0
    start_time = time.time()
    last_end_time = start_time
    optimizer_step = 0
    
    # 检查第一个batch的维度
    for features, data in dl:
        print(f"\n🔍 First batch dimensions:")
        print(f"   Features: {features.shape}")
        print(f"   Data: {data.shape}")
        
        # 移动数据到设备
        features = features.to(device, non_blocking=True)
        data = data.to(device, non_blocking=True)
        
        # 调整数据维度以匹配模型期望
        y = data[:, :N_samples]  # 裁剪到期望长度
        
        # 测试模型输出
        with torch.no_grad():
            if args.mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    y_hat = model(features[:, :T_frames, :], target_length=N_samples)
            else:
                y_hat = model(features[:, :T_frames, :], target_length=N_samples)
            print(f"   Model output: {y_hat.shape}")
            print(f"   Target: {y.shape}")
        
        print(f"✅ Dimension check passed!")
        print(f"💾 {get_memory_usage(device)}")
        break

    # 开始训练
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"{'='*50}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
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
                    optim.zero_grad(set_to_none=True)
                scaler.scale(loss_to_backward).backward()
                
                take_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
                if take_step:
                    scaler.step(optim)
                    scaler.update()
                    optimizer_step += 1
            else:
                y_hat = model(features[:, :T_frames, :], target_length=N_samples)
                loss = loss_fn(y_hat, y)
                loss_to_backward = loss / max(1, args.grad_accum_steps)
                
                if (batch_idx % args.grad_accum_steps) == 0:
                    optim.zero_grad(set_to_none=True)
                loss_to_backward.backward()
                
                take_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
                if take_step:
                    optim.step()
                    optimizer_step += 1
            
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # 统计计算耗时
            end_time = time.time()
            compute_time = end_time - now_time
            last_end_time = end_time
            
            if step % args.save_interval == 0:
                avg_loss = epoch_loss / num_batches
                elapsed_time = time.time() - start_time
                steps_per_sec = step / elapsed_time
                
                extra_timing = ''
                if args.log_timing:
                    extra_timing = f" | data {data_time*1000:.1f}ms, compute {compute_time*1000:.1f}ms"
                
                print(f'Epoch {epoch+1}/{args.epochs} Step {step} | '
                      f'L1 {loss.item():.6f} | Avg {avg_loss:.6f} | '
                      f'Speed: {steps_per_sec:.2f} steps/s | '
                      f'{get_memory_usage(device)}{extra_timing}')
                
                # 保存检查点
                ckpt = os.path.join(args.outdir, f'optfargan_{step}.pt')
                checkpoint = {
                    'step': step, 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'loss': loss.item(),
                    'device': str(device),
                    'mixed_precision': args.mixed_precision
                }
                if scaler is not None:
                    checkpoint['scaler'] = scaler.state_dict()
                
                torch.save(checkpoint, ckpt)
        
        # 轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        print(f'\n✅ Epoch {epoch+1} 完成 | 平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s | {get_memory_usage(device)}\n')

    # 训练完成
    total_time = time.time() - start_time
    print(f"🎉 训练完成!")
    print(f"📊 总耗时: {total_time/3600:.2f} 小时")
    print(f"⚡ 平均速度: {step/total_time:.2f} steps/s")
    print(f"💾 最终 {get_memory_usage(device)}")
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🗑️ GPU缓存已清理")


if __name__ == '__main__':
    main() 