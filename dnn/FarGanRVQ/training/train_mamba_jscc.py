#!/usr/bin/env python3
"""
MambaJSCC训练脚本 - 融合GSSM、CSI-ReST和信道自适应
基于MambaJSCC论文的完整实现
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
from dnn.FarGanRVQ.models.mamba_fargan_jscc import (
    MambaFarGanJSCC, MambaJSCCLoss
)

# 导入OSCE判别器
import sys
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../..", "torch", "osce"))
import models as osce_models


class ChannelSimulator:
    """信道仿真器，支持AWGN和瑞利衰落"""
    
    def __init__(self, channel_type='awgn'):
        self.channel_type = channel_type
        
    def __call__(self, x, snr_db):
        """
        Args:
            x: [B, ...] 输入信号
            snr_db: float 或 [B,] SNR (dB)
        Returns:
            noisy_x: [B, ...] 加噪信号
        """
        if self.channel_type == 'awgn':
            return self._awgn_channel(x, snr_db)
        elif self.channel_type == 'rayleigh':
            return self._rayleigh_channel(x, snr_db)
        else:
            return x
    
    def _awgn_channel(self, x, snr_db):
        """AWGN信道"""
        # 计算信号功率
        signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
        
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10.0)
        if isinstance(snr_linear, float):
            noise_power = signal_power / snr_linear
        else:
            snr_linear = snr_linear.view(-1, 1)
            noise_power = signal_power / snr_linear
        
        # 生成高斯噪声
        noise = torch.sqrt(noise_power) * torch.randn_like(x)
        
        return x + noise
    
    def _rayleigh_channel(self, x, snr_db):
        """瑞利衰落信道"""
        # 瑞利衰落系数
        h = (torch.randn_like(x) + 1j * torch.randn_like(x)) / np.sqrt(2)
        h_real = torch.abs(h)
        
        # 应用衰落
        x_faded = x * h_real
        
        # 添加AWGN
        return self._awgn_channel(x_faded, snr_db)


class MambaJSCCTrainer:
    """MambaJSCC训练器，支持CSI自适应和信道仿真"""
    
    def __init__(self, model, discriminators, optimizer_g, optimizer_d, 
                 loss_fn, writer, args, device):
        self.model = model
        self.discriminators = discriminators
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn
        self.writer = writer
        self.args = args
        self.device = device
        
        # 信道仿真器
        self.channel_sim = ChannelSimulator(args.channel_type)
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        
        # SNR范围
        self.snr_range = args.snr_range  # [min_snr, max_snr]
        
    def train_step(self, batch):
        """单步训练"""
        features, target = batch
        features = features.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        B = features.shape[0]
        
        # 随机采样SNR
        snr = torch.FloatTensor(B).uniform_(
            self.snr_range[0], self.snr_range[1]
        ).to(self.device)
        
        # 随机决定是否使用信道仿真
        use_channel = torch.rand(1).item() > 0.5
        
        # 生成音频
        if use_channel:
            # 模拟信道传输的噪声
            channel_noise = 'auto'  # 自动生成基于SNR的噪声
            generated = self.model(features, csi=snr, channel_noise=channel_noise)
        else:
            # 理想传输
            generated = self.model(features, csi=snr)
        
        # 调整目标长度
        min_len = min(generated.shape[1], target.shape[1])
        generated = generated[:, :min_len]
        target = target[:, :min_len]
        
        # ===== 更新判别器 =====
        for disc in self.discriminators:
            disc.train()
        
        self.optimizer_d.zero_grad()
        
        # 真实样本
        real_scores = []
        for disc in self.discriminators:
            real_score = disc(target.unsqueeze(1))
            real_scores.append(real_score)
        
        # 生成样本
        fake_scores = []
        for disc in self.discriminators:
            fake_score = disc(generated.detach().unsqueeze(1))
            fake_scores.append(fake_score)
        
        # 判别器损失
        d_loss = 0
        for real_score, fake_score in zip(real_scores, fake_scores):
            for scale_real, scale_fake in zip(real_score, fake_score):
                d_loss += F.mse_loss(scale_real[-1], torch.ones_like(scale_real[-1]))
                d_loss += F.mse_loss(scale_fake[-1], torch.zeros_like(scale_fake[-1]))
        
        d_loss = d_loss / (len(self.discriminators) * len(real_scores[0]))
        d_loss.backward()
        self.optimizer_d.step()
        
        # ===== 更新生成器 =====
        self.model.train()
        self.optimizer_g.zero_grad()
        
        # 重新计算生成样本的判别器分数
        gen_scores = []
        for disc in self.discriminators:
            gen_score = disc(generated.unsqueeze(1))
            gen_scores.append(gen_score)
        
        # 生成器损失
        g_losses = self.loss_fn(
            pred=generated,
            target=target,
            csi=snr,
            disc_real=real_scores,
            disc_fake=gen_scores
        )
        
        g_loss_total = g_losses['total']
        g_loss_total.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_g.step()
        
        # 记录损失
        losses = {
            'discriminator': d_loss.item(),
            'generator_total': g_loss_total.item(),
            **{k: v.item() if isinstance(v, torch.Tensor) else v 
               for k, v in g_losses.items() if k != 'total'}
        }
        
        return losses
    
    def log_step(self, losses, batch_size):
        """记录单步信息"""
        self.step += 1
        
        # TensorBoard记录
        for key, value in losses.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.step)
        
        # 记录SNR分布
        if self.step % 100 == 0:
            self.writer.add_histogram('Training/SNR_Distribution', 
                                    torch.FloatTensor(self.snr_range), self.step)
        
        # 性能统计
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.step / elapsed_time
        self.writer.add_scalar('Performance/Steps_Per_Second', steps_per_sec, self.step)
        
        # GPU内存使用
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            self.writer.add_scalar('System/GPU_Memory_GB', memory_used, self.step)
    
    def log_epoch(self, epoch_loss, epoch_time):
        """记录轮次信息"""
        self.epoch += 1
        
        self.writer.add_scalar('Training/Epoch_Loss', epoch_loss, self.epoch)
        self.writer.add_scalar('Performance/Epoch_Time_Minutes', epoch_time / 60, self.epoch)
        
        # 学习率记录
        self.writer.add_scalar('Training/LR_Generator', 
                             self.optimizer_g.param_groups[0]['lr'], self.epoch)
        self.writer.add_scalar('Training/LR_Discriminator', 
                             self.optimizer_d.param_groups[0]['lr'], self.epoch)


def collate_fn(batch):
    """数据加载函数"""
    features_list = []
    data_list = []
    
    for features, periods, data, lpc in batch:
        features_list.append(features)
        data_list.append(data)
    
    features = torch.from_numpy(np.array(features_list)).float()
    data = torch.from_numpy(np.array(data_list)).float()
    
    return features, data


def main():
    parser = argparse.ArgumentParser(description='MambaJSCC训练 - GSSM + CSI-ReST')
    parser.add_argument('--features', type=str, required=True, help='特征文件 .f32')
    parser.add_argument('--pcm', type=str, required=True, help='音频文件 .pcm')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-g', type=float, default=3e-5, help='生成器学习率')
    parser.add_argument('--lr-d', type=float, default=1e-4, help='判别器学习率')
    parser.add_argument('--seq-len', type=int, default=60, help='序列长度（帧）')
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--log-interval', type=int, default=10)
    
    # MambaJSCC特定参数
    parser.add_argument('--n-mamba-layers', type=int, default=4, help='Mamba层数')
    parser.add_argument('--cond-dim', type=int, default=128, help='条件维度')
    parser.add_argument('--channel-type', type=str, default='awgn', 
                       choices=['awgn', 'rayleigh'], help='信道类型')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20],
                       help='SNR范围 [min, max] (dB)')
    
    # 性能优化参数
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--compile', action='store_true')
    
    args = parser.parse_args()
    
    print(f"🎯 MambaJSCC训练启动")
    print(f"{'='*50}")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 训练设备: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"⚡ GPU优化已启用")
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'mamba_jscc_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard: {log_dir}")
    
    # 数据加载
    frame_size = 160
    seq_len = args.seq_len
    F_used = 20
    
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
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True
    )
    
    print(f"\n📋 MambaJSCC配置:")
    print(f"  ├ Mamba层数: {args.n_mamba_layers}")
    print(f"  ├ 条件维度: {args.cond_dim}")
    print(f"  ├ 信道类型: {args.channel_type}")
    print(f"  ├ SNR范围: {args.snr_range} dB")
    print(f"  ├ 序列长度: {seq_len} 帧")
    print(f"  └ 批量大小: {args.batch_size}")
    
    # 模型初始化
    model = MambaFarGanJSCC(
        in_features=F_used,
        cond_dim=args.cond_dim,
        n_mamba_layers=args.n_mamba_layers
    ).to(device)
    
    # 判别器
    discriminators = [
        osce_models.model_dict['fdmresdisc'](
            architecture='free', design='f_down',
            fft_sizes_16k=[2**n for n in range(6, 12)], 
            freq_roi=[0, 7400], max_channels=256, noise_gain=0.0
        ).to(device),
        osce_models.model_dict['fdmresdisc'](
            architecture='free', design='f_down',
            fft_sizes_16k=[2**n for n in range(7, 11)],
            freq_roi=[0, 8000], max_channels=128, noise_gain=0.1
        ).to(device)
    ]
    
    # 损失函数
    loss_fn = MambaJSCCLoss(
        spectral_weight=1.0,
        adversarial_weight=0.1,
        channel_weight=0.05
    ).to(device)
    
    # 优化器
    optimizer_g = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_g,
        betas=(0.8, 0.99),
        weight_decay=1e-4
    )
    
    optimizer_d = torch.optim.AdamW(
        [p for disc in discriminators for p in disc.parameters()],
        lr=args.lr_d,
        betas=(0.8, 0.99)
    )
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_g, T_0=5000, T_mult=2, eta_min=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_d, T_0=5000, T_mult=2, eta_min=1e-6
    )
    
    # 混合精度
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    
    # 模型编译
    if args.compile and hasattr(torch, 'compile'):
        print(f"🔥 启用模型编译...")
        model = torch.compile(model)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数: {total_params:,}")
    
    # 记录超参数
    writer.add_hparams({
        'batch_size': args.batch_size,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'n_mamba_layers': args.n_mamba_layers,
        'cond_dim': args.cond_dim,
        'channel_type': args.channel_type,
        'snr_min': args.snr_range[0],
        'snr_max': args.snr_range[1],
        'model_params': total_params
    }, {})
    
    # 创建训练器
    trainer = MambaJSCCTrainer(
        model=model,
        discriminators=discriminators,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        loss_fn=loss_fn,
        writer=writer,
        args=args,
        device=device
    )
    
    print(f"\n🚀 开始MambaJSCC训练...")
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        model.train()
        for disc in discriminators:
            disc.train()
        
        for batch_idx, batch in enumerate(dl):
            # 训练步骤
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    losses = trainer.train_step(batch)
            else:
                losses = trainer.train_step(batch)
            
            epoch_losses.append(losses['generator_total'])
            
            # 记录
            trainer.log_step(losses, batch[0].shape[0])
            
            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()
            
            # 打印进度
            if trainer.step % args.log_interval == 0:
                elapsed_time = time.time() - trainer.start_time
                steps_per_sec = trainer.step / elapsed_time
                
                mem_info = ""
                if device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_info = f"GPU: {mem_used:.2f}GB"
                
                print(f'Epoch {epoch+1}/{args.epochs} Step {trainer.step} | '
                      f'G_loss: {losses["generator_total"]:.6f} | '
                      f'D_loss: {losses["discriminator"]:.6f} | '
                      f'LR_G: {optimizer_g.param_groups[0]["lr"]:.2e} | '
                      f'Speed: {steps_per_sec:.2f} steps/s | {mem_info}')
        
        # 轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(epoch_losses)
        trainer.log_epoch(avg_epoch_loss, epoch_time)
        
        print(f'✅ Epoch {epoch+1} 完成 | 平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s')
        
        # 保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args)
            }
            checkpoint_path = os.path.join(log_dir, f'mamba_jscc_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    # 训练完成
    total_time = time.time() - trainer.start_time
    writer.add_text('Training/Completion', f"""
    **MambaJSCC训练完成!**
    - 总步数: {trainer.step:,}
    - 总时长: {total_time/3600:.2f} 小时
    - 平均速度: {trainer.step/total_time:.2f} steps/s
    - 模型参数: {total_params:,}
    """)
    
    writer.close()
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\n🎉 MambaJSCC训练完成!")
    print(f"📊 查看结果: tensorboard --logdir={args.log_dir}")
    print(f"⚡ 平均速度: {trainer.step/total_time:.2f} steps/s")


if __name__ == '__main__':
    main() 