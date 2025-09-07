#!/usr/bin/env python3
"""
FarGanRVQ 高 GPU 利用率训练脚本
专门针对提升GPU利用率进行优化
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
from dnn.FarGanRVQ.models.larger_fargan import LargerFarGan


def setup_device():
    """设置训练设备并显示相关信息"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 使用GPU训练: {gpu_name}")
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
        
        # GPU 优化设置
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f"🗑️ GPU缓存已清理")
        print(f"⚡ cuDNN benchmark已启用")
        print(f"⚙️ TF32 已启用")
        
        return device, gpu_name, gpu_memory
    else:
        device = torch.device('cpu')
        print(f"⚠️ 未检测到GPU，使用CPU训练")
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


def main():
    parser = argparse.ArgumentParser(description='FarGanRVQ 高GPU利用率训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件 .f32')
    parser.add_argument('--pcm', type=str, required=True, help='音频文件 .pcm')
    parser.add_argument('--batch-size', type=int, default=4096, help='超大批量以提升GPU利用率')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq-len', type=int, default=20, help='更长序列提升计算量')
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--model-size', type=str, default='large', choices=['small', 'large'], 
                       help='模型大小: small=65万参数, large=300万+参数')
    
    # 强制启用所有GPU优化
    parser.add_argument('--num-workers', type=int, default=12, help='更多数据加载线程')
    parser.add_argument('--prefetch-factor', type=int, default=8, help='更大预取因子')
    parser.add_argument('--grad-accum-steps', type=int, default=8, help='大梯度累积')
    
    args = parser.parse_args()
    
    print(f"🎯 FarGanRVQ 高GPU利用率训练启动")
    print(f"{'='*60}")
    
    # 设置设备
    device, gpu_name, gpu_memory = setup_device()
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'high_gpu_{args.model_size}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    
    # 数据集设置
    frame_size = 160
    seq_len = args.seq_len
    T_frames = seq_len * 2 + 4
    F_used = 20
    N_samples = frame_size * seq_len
    
    print(f"\n📋 高性能训练配置:")
    print(f"  ├ 模型大小: {args.model_size}")
    print(f"  ├ 批量大小: {args.batch_size}")
    print(f"  ├ 序列长度: {seq_len} 帧")
    print(f"  ├ 学习率: {args.lr}")
    print(f"  ├ 输出样本数: {N_samples}")
    print(f"  ├ 数据线程: {args.num_workers}")
    print(f"  ├ 预取因子: {args.prefetch_factor}")
    print(f"  ├ 梯度累积: {args.grad_accum_steps} 步")
    print(f"  └ 训练设备: {device}")
    
    # 高性能数据加载器
    pin_memory = True
    persistent_workers = True
    
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
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        drop_last=True,
        prefetch_factor=args.prefetch_factor
    )
    
    # 选择模型
    if args.model_size == 'large':
        model = LargerFarGan(in_features=F_used, cond_dim=192)
    else:
        model = OptimizedFarGan(in_features=F_used, cond_dim=96)
    
    model = model.to(device)
    
    # 模型编译优化
    if hasattr(torch, 'compile'):
        print(f"🔥 启用模型编译优化...")
        model = torch.compile(model, mode='max-autotune')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: {trainable_params:,} (可训练) / {total_params:,} (总计)")
    print(f"💾 模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 记录超参数
    writer.add_hparams({
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seq_len': seq_len,
        'model_params': total_params,
        'grad_accum_steps': args.grad_accum_steps,
        'num_workers': args.num_workers,
    }, {})
    
    print(f"\n🚀 开始高强度训练...")
    
    model.train()
    step = 0
    start_time = time.time()
    last_end_time = start_time
    
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
            with torch.amp.autocast('cuda'):
                y_hat = model(features[:, :T_frames, :], target_length=N_samples)
                loss = loss_fn(y_hat, y)
                loss_to_backward = loss / args.grad_accum_steps
            
            if (batch_idx % args.grad_accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            
            scaler.scale(loss_to_backward).backward()
            
            take_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
            if take_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            step += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            # 统计计算耗时
            end_time = time.time()
            compute_time = end_time - now_time
            last_end_time = end_time
            
            # 记录指标
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('Performance/Data_Load_Time_s', data_time, step)
            writer.add_scalar('Performance/Compute_Time_s', compute_time, step)
            
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                writer.add_scalar('System/GPU_Memory_Used_GB', memory_used, step)
                writer.add_scalar('System/GPU_Memory_Cached_GB', memory_cached, step)
            
            # 打印进度
            if step % args.log_interval == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = step / elapsed_time
                
                if device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_cached = torch.cuda.memory_reserved() / 1024**3
                    mem_info = f"GPU: {mem_used:.2f}GB used, {mem_cached:.2f}GB cached"
                else:
                    import psutil
                    mem_info = f"RAM: {psutil.virtual_memory().percent:.1f}%"
                
                print(f'Epoch {epoch+1}/{args.epochs} Step {step} | '
                      f'Loss: {loss.item():.6f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                      f'Speed: {steps_per_sec:.2f} steps/s | '
                      f'{mem_info} | '
                      f'data {data_time*1000:.1f}ms, compute {compute_time*1000:.1f}ms')
        
        # 记录轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        writer.add_scalar('Training/Epoch_Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Performance/Epoch_Time_Minutes', epoch_time / 60, epoch)
        
        print(f'✅ Epoch {epoch+1} 完成 | 平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s')
        
        # 保存检查点
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args)
            }
            checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 已保存检查点: {checkpoint_path}")
    
    # 训练完成
    total_time = time.time() - start_time
    writer.add_text('Training/Completion', f"""
    **高GPU利用率训练完成!**
    - 总步数: {step:,}
    - 总时长: {total_time/3600:.2f} 小时
    - 平均速度: {step/total_time:.2f} steps/s
    - 模型参数: {total_params:,}
    - 最大GPU使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB
    """)
    
    writer.close()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    print(f"🗑️ GPU缓存已清理")
    
    print(f"\n🎉 高GPU利用率训练完成!")
    print(f"📊 查看结果: tensorboard --logdir={args.log_dir}")
    print(f"⚡ 平均速度: {step/total_time:.2f} steps/s")
    print(f"💾 最大GPU使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    main() 