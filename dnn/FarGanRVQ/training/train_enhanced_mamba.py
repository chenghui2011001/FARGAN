#!/usr/bin/env python3
"""
增强的FarGanRVQ对抗训练 - 融合MambaJSCC技术
基于现有enhanced_adv_train.py，添加GSSM + CSI-ReST支持
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
fargan_path = os.path.join(project_root, 'dnn/torch/fargan')
sys.path.insert(0, fargan_path)

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.mamba_enhanced_fargan import (
    MambaEnhancedFarGan, MambaJSCCEnhancedLoss
)
from dnn.FarGanRVQ.models.enhanced_losses import (
    PitchConsistencyLoss, PhaseAwareSpectralLoss, PerceptualLoss
)

# 导入OSCE判别器
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../..", "torch", "osce"))
import models as osce_models


def collate_fn(batch):
    """高效数据加载函数"""
    features_list = []
    data_list = []
    
    for features, periods, data, lpc in batch:
        features_list.append(features)
        data_list.append(data)
    
    features = torch.from_numpy(np.array(features_list)).float()
    data = torch.from_numpy(np.array(data_list)).float()
    
    return features, data


def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 使用GPU: {gpu_name}")
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
        
        # GPU优化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return device, gpu_name, gpu_memory
    else:
        device = torch.device('cpu')
        print(f"⚠️ 使用CPU训练")
        return device, "CPU", 0


class MambaJSCCTrainer:
    """MambaJSCC训练器，基于现有框架"""
    
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
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        
        # 学习率调度器
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_g, T_0=5000, T_mult=2, eta_min=1e-6
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_d, T_0=5000, T_mult=2, eta_min=1e-6
        )
        
        # SNR范围
        self.snr_range = args.snr_range
        
    def train_step(self, batch):
        """单步训练，支持CSI和信道仿真"""
        features, target = batch
        features = features.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        B = features.shape[0]
        
        # 随机采样SNR（如果启用）
        csi = None
        channel_noise = None
        if self.args.enable_csi:
            csi = torch.FloatTensor(B).uniform_(
                self.snr_range[0], self.snr_range[1]
            ).to(self.device)
            
            # 随机决定是否使用信道仿真
            if torch.rand(1).item() < self.args.channel_prob:
                channel_noise = 'auto'
        
        # 生成音频
        N_samples = min(target.shape[1], features.shape[1] * 160)
        generated = self.model(
            features,
            csi=csi,
            channel_noise=channel_noise,
            target_length=N_samples,
            parallel_train=self.args.parallel_train,
            teacher_signal=target if self.args.parallel_train else None,
        )
        
        # 调整长度
        min_len = min(generated.shape[1], target.shape[1])
        generated = generated[:, :min_len]
        target = target[:, :min_len]
        
        # ===== 更新判别器（按频率） =====
        has_disc = len(self.discriminators) > 0
        update_d = has_disc and ((self.args.adv_every <= 1) or (self.step % self.args.adv_every == 0))
        d_loss_value = 0.0
        real_scores_all = None
        if update_d:
            self.optimizer_d.zero_grad()
            d_loss = 0
            real_scores_all = []
            fake_scores_all = []
            autocast_ctx = torch.amp.autocast('cuda') if self.args.mixed_precision else contextlib.nullcontext()
            with autocast_ctx:
                for disc in self.discriminators:
                    disc.train()
                    real_scores = disc(target.unsqueeze(1))
                    real_scores_all.append(real_scores)
                    fake_scores = disc(generated.detach().unsqueeze(1))
                    fake_scores_all.append(fake_scores)
                for real_score, fake_score in zip(real_scores, fake_scores):
                    d_loss += F.mse_loss(real_score[-1], torch.ones_like(real_score[-1]))
                    d_loss += F.mse_loss(fake_score[-1], torch.zeros_like(fake_score[-1]))
            d_loss = d_loss / (len(self.discriminators) * len(real_scores_all[0]))
            if self.args.mixed_precision:
                with torch.amp.autocast('cuda'):
                    d_loss.backward()
            else:
                d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for disc in self.discriminators for p in disc.parameters()], 
                max_norm=1.0
            )
            self.optimizer_d.step()
            d_loss_value = float(d_loss.item())
        
        # ===== 更新生成器 =====
        self.model.train()
        self.optimizer_g.zero_grad()
        
        # 仅在需要时计算生成样本的判别器分数（否则跳过对抗项）
        gen_scores_all = None
        if update_d:
            gen_scores_all = []
            autocast_ctx = torch.amp.autocast('cuda') if self.args.mixed_precision else contextlib.nullcontext()
            with autocast_ctx:
                for disc in self.discriminators:
                    gen_scores = disc(generated.unsqueeze(1))
                    gen_scores_all.append(gen_scores)
        
        # 生成器损失
        if self.args.mixed_precision:
            with torch.amp.autocast('cuda'):
                g_losses = self.loss_fn(
                    pred=generated,
                    target=target,
                    csi=csi,
                    disc_real=real_scores_all,
                    disc_fake=gen_scores_all
                )
        else:
            g_losses = self.loss_fn(
                pred=generated,
                target=target,
                csi=csi,
                disc_real=real_scores_all,
                disc_fake=gen_scores_all
            )
        
        g_loss_total = g_losses['total']
        g_loss_total.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_g.step()
        
        # 更新学习率
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        # 记录损失
        losses = {
            'discriminator': d_loss_value,
            'generator_total': g_loss_total.item(),
            **{k: v.item() if isinstance(v, torch.Tensor) else v 
               for k, v in g_losses.items() if k != 'total'}
        }
        
        # 记录CSI信息
        if csi is not None:
            losses['avg_snr'] = csi.mean().item()
            losses['channel_used'] = 1.0 if channel_noise is not None else 0.0
        
        return losses
    
    def log_step(self, losses, batch_size):
        """记录单步信息"""
        self.step += 1
        
        # TensorBoard记录
        for key, value in losses.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.step)
        
        # 学习率记录
        self.writer.add_scalar('Training/LR_Generator', 
                             self.optimizer_g.param_groups[0]['lr'], self.step)
        self.writer.add_scalar('Training/LR_Discriminator', 
                             self.optimizer_d.param_groups[0]['lr'], self.step)
        
        # 性能统计
        elapsed_time = time.time() - self.start_time
        steps_per_sec = self.step / elapsed_time
        self.writer.add_scalar('Performance/Steps_Per_Second', steps_per_sec, self.step)
        
        # GPU内存
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            self.writer.add_scalar('System/GPU_Memory_GB', memory_used, self.step)


def main():
    parser = argparse.ArgumentParser(description='MambaJSCC增强对抗训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件')
    parser.add_argument('--pcm', type=str, required=True, help='音频文件')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-g', type=float, default=3e-5)
    parser.add_argument('--lr-d', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--outdir', type=str, default=None, help='日志与检查点输出根目录（优先于 --log-dir）')
    parser.add_argument('--log-interval', type=int, default=10)
    
    # MambaJSCC特定参数
    parser.add_argument('--enable-csi', action='store_true', 
                       help='启用CSI-ReST信道自适应')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20],
                       help='SNR范围 [min, max] (dB)')
    parser.add_argument('--channel-prob', type=float, default=0.5,
                       help='信道仿真概率')
    parser.add_argument('--n-mamba-layers', type=int, default=4,
                       help='Mamba层数')
    parser.add_argument('--parallel-train', action='store_true',
                       help='启用训练期并行子帧（Teacher Forcing）')
    parser.add_argument('--compile', action='store_true',
                       help='使用 torch.compile 编译模型以提升吞吐')
    parser.add_argument('--adv-every', type=int, default=1,
                       help='每隔多少步更新一次判别器/参与对抗损失 (默认每步)')
    parser.add_argument('--disable-adv', action='store_true',
                       help='禁用对抗分支和判别器，便于快速启动或省显存')
    parser.add_argument('--disc1-max-ch', type=int, default=256,
                       help='判别器1的最大通道数')
    parser.add_argument('--disc2-max-ch', type=int, default=128,
                       help='判别器2的最大通道数')
    parser.add_argument('--max-steps-per-epoch', type=int, default=0,
                       help='限制每个epoch的训练步数(0为不限)')
    parser.add_argument('--stft-sizes', type=int, nargs='+', default=[512, 1024, 2048],
                       help='多分辨率STFT的fft尺寸列表')
    parser.add_argument('--disable-phase-loss', action='store_true',
                       help='禁用相位损失(将phase_weight设为0)')
    
    # 原有的性能参数
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--drop-last', action='store_true', default=True)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--enable-tf32', action='store_true')
    parser.add_argument('--log-timing', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true')
    
    args = parser.parse_args()
    
    print(f"🎯 MambaJSCC增强训练启动")
    print(f"{'='*50}")
    
    # 设备设置
    device, gpu_name, gpu_memory = setup_device()
    
    # 日志/输出目录（优先使用 --outdir）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = args.outdir if args.outdir is not None else args.log_dir
    log_dir = os.path.join(base_dir, f'mamba_enhanced_{timestamp}')
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
    
    # 数据加载器优化
    num_workers = args.num_workers
    pin_memory = device.type == 'cuda'
    persistent_workers = num_workers > 0
    
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
    
    print(f"\n📋 MambaJSCC配置:")
    print(f"  ├ 启用CSI: {args.enable_csi}")
    if args.enable_csi:
        print(f"  ├ SNR范围: {args.snr_range} dB")
        print(f"  ├ 信道概率: {args.channel_prob}")
    print(f"  ├ Mamba层数: {args.n_mamba_layers}")
    print(f"  ├ 序列长度: {seq_len} 帧")
    print(f"  ├ 批量大小: {args.batch_size}")
    print(f"  └ 混合精度: {args.mixed_precision}")
    
    # 模型初始化
    model = MambaEnhancedFarGan(
        in_features=F_used,
        cond_dim=32,
        subframe_size=40
    ).to(device)

    # 可选编译
    if args.compile and hasattr(torch, 'compile'):
        print("🔥 启用 torch.compile 优化...")
        model = torch.compile(model, mode='max-autotune')
    
    # 判别器
    discriminators = []
    if not args.disable_adv:
        discriminators = [
            osce_models.model_dict['fdmresdisc'](
                architecture='free', design='f_down',
                fft_sizes_16k=[2**n for n in range(6, 12)], 
                freq_roi=[0, 7400], max_channels=args.disc1_max_ch, noise_gain=0.0
            ).to(device),
            osce_models.model_dict['fdmresdisc'](
                architecture='free', design='f_down',
                fft_sizes_16k=[2**n for n in range(7, 11)],
                freq_roi=[0, 8000], max_channels=args.disc2_max_ch, noise_gain=0.1
            ).to(device)
        ]
    
    # 损失函数
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    loss_fn = MambaJSCCEnhancedLoss(
        spectral_weight=1.0,
        adversarial_weight=0.1,
        channel_weight=0.05 if args.enable_csi else 0.0,
        phase_weight=phase_weight,
        stft_sizes=args.stft_sizes
    )
    
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
    ) if len(discriminators) > 0 else torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-8)
    
    # 混合精度
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数: {total_params:,}")
    
    # 记录超参数
    writer.add_hparams({
        'batch_size': args.batch_size,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'seq_len': seq_len,
        'enable_csi': args.enable_csi,
        'snr_min': args.snr_range[0] if args.enable_csi else 0,
        'snr_max': args.snr_range[1] if args.enable_csi else 0,
        'channel_prob': args.channel_prob,
        'n_mamba_layers': args.n_mamba_layers,
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
    
    print(f"\n🚀 开始MambaJSCC增强训练...")
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # 训练模式
        model.train()
        for disc in discriminators:
            disc.train()
        
        # 进度条
        tepoch = tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, batch in enumerate(tepoch):
            # 统计数据加载时间
            data_start_time = time.time()
            
            # 训练步骤
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    losses = trainer.train_step(batch)
            else:
                losses = trainer.train_step(batch)
            
            # 统计计算时间
            compute_time = time.time() - data_start_time
            
            epoch_losses.append(losses['generator_total'])
            
            # 记录
            trainer.log_step(losses, batch[0].shape[0])
            
            # 更新进度条
            postfix_dict = {
                'G_loss': f"{losses['generator_total']:.4f}",
                'D_loss': f"{losses['discriminator']:.4f}",
                'LR_G': f"{optimizer_g.param_groups[0]['lr']:.2e}",
            }
            
            if args.enable_csi and 'avg_snr' in losses:
                postfix_dict['SNR'] = f"{losses['avg_snr']:.1f}dB"
                postfix_dict['CH'] = f"{losses['channel_used']:.0f}"
            
            if args.log_timing:
                postfix_dict['Time'] = f"{compute_time*1000:.0f}ms"
            
            tepoch.set_postfix(postfix_dict)
            
            # 打印详细进度
            if trainer.step % args.log_interval == 0:
                elapsed_time = time.time() - trainer.start_time
                steps_per_sec = trainer.step / elapsed_time
                
                mem_info = ""
                if device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_info = f"GPU: {mem_used:.2f}GB"
                
                csi_info = ""
                if args.enable_csi and 'avg_snr' in losses:
                    csi_info = f"SNR: {losses['avg_snr']:.1f}dB"
                
                print(f'\nStep {trainer.step} | '
                      f'G: {losses["generator_total"]:.6f} | '
                      f'D: {losses["discriminator"]:.6f} | '
                      f'Speed: {steps_per_sec:.2f} steps/s | '
                      f'{mem_info} {csi_info}')

            # 限制每个epoch的最大步数
            if args.max_steps_per_epoch and (batch_idx + 1) >= args.max_steps_per_epoch:
                break

        
        
        # 轮次总结
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(epoch_losses)
        
        print(f'\n✅ Epoch {epoch+1} 完成 | '
              f'平均损失: {avg_epoch_loss:.6f} | '
              f'耗时: {epoch_time:.1f}s')
        
        # 保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'loss': avg_epoch_loss,
                'args': vars(args)
            }
            checkpoint_path = os.path.join(log_dir, f'mamba_enhanced_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")
    
    # 训练完成
    total_time = time.time() - trainer.start_time
    
    writer.add_text('Training/Completion', f"""
    **MambaJSCC增强训练完成!**
    - 总步数: {trainer.step:,}
    - 总时长: {total_time/3600:.2f} 小时
    - 平均速度: {trainer.step/total_time:.2f} steps/s
    - 模型参数: {total_params:,}
    - CSI支持: {args.enable_csi}
    - Mamba层数: {args.n_mamba_layers}
    """)
    
    writer.close()
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\n🎉 MambaJSCC增强训练完成!")
    print(f"📊 查看结果: tensorboard --logdir={args.log_dir}")
    print(f"⚡ 平均速度: {trainer.step/total_time:.2f} steps/s")
    
    # 显示关键特性
    print(f"\n🔬 MambaJSCC核心特性:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✓ GSSM双向扫描: 全局信息捕获")
    print(f"✓ CSI-ReST机制: 零参数信道自适应")
    print(f"✓ 相位感知损失: 提升音质一致性")
    print(f"✓ 自适应损失权重: 信道条件感知")


if __name__ == '__main__':
    main() 
