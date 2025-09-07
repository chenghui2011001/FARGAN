#!/usr/bin/env python3
"""
MambaEnhancedFarGan 对抗微调脚本（含判别器）
- 从预训练检查点恢复生成器 
- 使用 OSCE 频域多分辨率判别器进行 LSGAN + Feature Matching（由 MambaJSCCEnhancedLoss 支持）
"""

import argparse
import os
import sys
import time
from datetime import datetime
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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

# OSCE 判别器
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../..", "torch", "osce"))
import models as osce_models


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
    return device


def collate_fn(batch):
    feats, data = [], []
    for features, periods, waveform, lpc in batch:
        feats.append(features)
        data.append(waveform)
    feats = torch.from_numpy(np.array(feats)).float()
    data = torch.from_numpy(np.array(data)).float()
    return feats, data


def main():
    parser = argparse.ArgumentParser(description='MambaEnhancedFarGan 对抗微调')
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--pcm', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True, help='预训练检查点路径(.pth)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-g', type=float, default=3e-5)
    parser.add_argument('--lr-d', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=20)

    # 功能/性能
    parser.add_argument('--parallel-train', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=6)
    parser.add_argument('--drop-last', action='store_true', default=True)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0)
    parser.add_argument('--adv-every', type=int, default=1, help='每N步更新一次判别器并计算对抗项')
    parser.add_argument('--disc1-max-ch', type=int, default=256)
    parser.add_argument('--disc2-max-ch', type=int, default=128)

    # 损失/信道
    parser.add_argument('--enable-csi', action='store_true')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20])
    parser.add_argument('--channel-prob', type=float, default=0.5)
    parser.add_argument('--stft-sizes', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--disable-phase-loss', action='store_true')

    args = parser.parse_args()

    device = setup_device()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = args.outdir if args.outdir is not None else args.log_dir
    log_dir = os.path.join(base_dir, f'adv_mamba_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 数据
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
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=args.drop_last,
        collate_fn=collate_fn,
    )

    # 模型与恢复
    model = MambaEnhancedFarGan(in_features=F_used, cond_dim=32, subframe_size=40).to(device)
    if os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        msd = ckpt.get('model_state_dict', None)
        if msd is not None:
            model.load_state_dict(msd, strict=False)
            print(f'🔁 已载入预训练: {args.resume}')
    if args.compile and hasattr(torch, 'compile'):
        print('🔥 compile model (inductor)')
        model = torch.compile(model, mode='max-autotune')

    # 判别器
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

    # 损失
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    loss_fn = MambaJSCCEnhancedLoss(
        spectral_weight=1.0,
        adversarial_weight=0.1,
        channel_weight=0.05 if args.enable_csi else 0.0,
        phase_weight=phase_weight,
        stft_sizes=args.stft_sizes
    )

    # 优化器
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW([p for d in discriminators for p in d.parameters()], lr=args.lr_d, betas=(0.8, 0.99))
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

    # 训练
    step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        [d.train() for d in discriminators]
        tepoch = tqdm(dl, desc=f'Adv Epoch {epoch+1}/{args.epochs}')
        for bidx, (features, target) in enumerate(tepoch):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 随机CSI/信道
            csi = None
            channel_noise = None
            if args.enable_csi:
                B = features.shape[0]
                csi = torch.FloatTensor(B).uniform_(args.snr_range[0], args.snr_range[1]).to(device)
                if torch.rand(1).item() < args.channel_prob:
                    channel_noise = 'auto'

            N_samples = min(target.shape[1], features.shape[1] * 160)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                y_hat = model(
                    features,
                    csi=csi,
                    channel_noise=channel_noise,
                    target_length=N_samples,
                    parallel_train=args.parallel_train,
                    teacher_signal=target if args.parallel_train else None,
                )
                min_len = min(y_hat.shape[1], target.shape[1])
                y_hat = y_hat[:, :min_len]
                target_ = target[:, :min_len]

            # 更新判别器（按频率）
            update_d = (args.adv_every <= 1) or (step % args.adv_every == 0)
            d_loss_value = 0.0
            real_scores_all = None
            if update_d:
                optimizer_d.zero_grad(set_to_none=True)
                d_loss = 0.0
                real_scores_all = []
                fake_scores_all = []
                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    for disc in discriminators:
                        real_scores = disc(target_.unsqueeze(1))
                        real_scores_all.append(real_scores)
                        fake_scores = disc(y_hat.detach().unsqueeze(1))
                        fake_scores_all.append(fake_scores)
                        for r_s, f_s in zip(real_scores, fake_scores):
                            d_loss = d_loss + F.mse_loss(r_s[-1], torch.ones_like(r_s[-1]))
                            d_loss = d_loss + F.mse_loss(f_s[-1], torch.zeros_like(f_s[-1]))
                    d_loss = d_loss / (len(discriminators) * len(real_scores_all[0]))

                if scaler is not None:
                    scaler.scale(d_loss).backward()
                    for d in discriminators:
                        torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=1.0)
                    scaler.step(optimizer_d)
                else:
                    d_loss.backward()
                    for d in discriminators:
                        torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=1.0)
                    optimizer_d.step()
                d_loss_value = float(d_loss.item())

            # 生成器更新
            optimizer_g.zero_grad(set_to_none=True)
            gen_scores_all = None
            if update_d:
                gen_scores_all = []
                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    for disc in discriminators:
                        gen_scores = disc(y_hat.unsqueeze(1))
                        gen_scores_all.append(gen_scores)

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                g_losses = loss_fn(
                    pred=y_hat,
                    target=target_,
                    csi=csi,
                    disc_real=real_scores_all,
                    disc_fake=gen_scores_all,
                )
                g_total = g_losses['total']

            if scaler is not None:
                scaler.scale(g_total).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_g)
                scaler.update()
            else:
                g_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_g.step()

            step += 1
            tepoch.set_postfix({'G': f'{float(g_total.item()):.4f}', 'D': f'{d_loss_value:.4f}'})

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-6)
                writer.add_scalar('Loss/adv_total', float(g_total.item()), step)
                writer.add_scalar('Loss/discriminator', d_loss_value, step)
                writer.add_scalar('Perf/steps_per_sec', sps, step)

            if args.max_steps_per_epoch and (bidx + 1) >= args.max_steps_per_epoch:
                break

        # 保存
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'args': vars(args),
            }
            ckpt_path = os.path.join(log_dir, f'adv_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f'💾 保存对抗检查点: {ckpt_path}')

    writer.close()


if __name__ == '__main__':
    main()

