#!/usr/bin/env python3
"""
MambaEnhancedFarGan 预训练脚本（无对抗分支）
- 仅使用重建类损失（多分辨率STFT/相位/信道自适应）
- 支持并行子帧（Teacher Forcing）、AMP、torch.compile、步数限制与快速I/O设置
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

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
    parser = argparse.ArgumentParser(description='MambaEnhancedFarGan 预训练（无对抗）')
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--pcm', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=20, help='每N个epoch保存一次模型')

    # 性能与功能
    parser.add_argument('--parallel-train', action='store_true', help='启用并行子帧 TF')
    parser.add_argument('--compile', action='store_true', help='torch.compile')
    parser.add_argument('--mixed-precision', action='store_true', help='AMP 训练')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=6)
    parser.add_argument('--drop-last', action='store_true', default=True)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0)

    # 损失相关
    parser.add_argument('--enable-csi', action='store_true')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20])
    parser.add_argument('--channel-prob', type=float, default=0.5)
    parser.add_argument('--stft-sizes', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--disable-phase-loss', action='store_true')

    args = parser.parse_args()

    device = setup_device()

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = args.outdir if args.outdir is not None else args.log_dir
    log_dir = os.path.join(base_dir, f'pretrain_mamba_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 数据集与加载器
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

    # 模型
    model = MambaEnhancedFarGan(in_features=F_used, cond_dim=32, subframe_size=40).to(device)
    if args.compile and hasattr(torch, 'compile'):
        print('🔥 compile model (inductor)')
        model = torch.compile(model, mode='max-autotune')

    # 损失（无对抗分支）
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    loss_fn = MambaJSCCEnhancedLoss(
        spectral_weight=1.0,
        adversarial_weight=0.0,  # 关闭对抗
        channel_weight=0.05 if args.enable_csi else 0.0,
        phase_weight=phase_weight,
        stft_sizes=args.stft_sizes,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

    total_params = sum(p.numel() for p in model.parameters())
    writer.add_hparams({
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seq_len': seq_len,
        'model_params': total_params,
        'parallel_train': args.parallel_train,
        'stft_sizes': str(args.stft_sizes),
    }, {})

    step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        tepoch = tqdm(dl, desc=f'Pretrain Epoch {epoch+1}/{args.epochs}')
        for bidx, (features, target) in enumerate(tepoch):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            N_samples = min(target.shape[1], features.shape[1] * 160)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                y_hat = model(
                    features,
                    csi=None,
                    channel_noise=None,
                    target_length=N_samples,
                    parallel_train=args.parallel_train,
                    teacher_signal=target if args.parallel_train else None,
                )

                # 截断对齐
                min_len = min(y_hat.shape[1], target.shape[1])
                y_hat = y_hat[:, :min_len]
                target_ = target[:, :min_len]

                losses = loss_fn(pred=y_hat, target=target_, csi=None, disc_real=None, disc_fake=None)
                loss = losses['total']

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            step += 1
            epoch_losses.append(float(loss.item()))
            tepoch.set_postfix({'L': f'{epoch_losses[-1]:.4f}'})

            # 日志
            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-6)
                writer.add_scalar('Loss/pretrain_total', float(loss.item()), step)
                writer.add_scalar('Perf/steps_per_sec', sps, step)

            # 限制步数
            if args.max_steps_per_epoch and (bidx + 1) >= args.max_steps_per_epoch:
                break

        # 保存
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }
            ckpt_path = os.path.join(log_dir, f'pretrain_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f'💾 保存预训练检查点: {ckpt_path}')

    writer.close()


if __name__ == '__main__':
    main()

