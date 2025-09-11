#!/usr/bin/env python3
"""
MambaEnhancedFarGan 对抗微调脚本（含判别器, 支持单机多卡 DDP）
- 从预训练检查点恢复生成器
- 使用 OSCE 频域多分辨率判别器进行 LSGAN + Feature Matching（由 MambaJSCCEnhancedLoss 支持）
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# -------------------------- 路径 --------------------------
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
import models as osce_models  # noqa


# -------------------------- 分布式/设备工具 --------------------------
def ddp_env():
    """返回 (is_distributed, rank, world_size, local_rank)"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def setup_distributed():
    is_dist, rank, world, local = ddp_env()
    if is_dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return is_dist, rank, world, local


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_device(local_rank: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')  # 或 'medium'
        except Exception:
            pass
    else:
        device = torch.device('cpu')
    return device


def iter_microbatches(*tensors, microbatch: int):
    """按 batch 维切分若干块，返回每块的张量元组"""
    if microbatch is None or microbatch <= 0:
        yield tensors
        return
    B = tensors[0].shape[0]
    for i in range(0, B, microbatch):
        sl = slice(i, min(i + microbatch, B))
        yield tuple(t[sl] if t is not None else None for t in tensors)


# -------------------------- dataloader --------------------------
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

    # 微批 & DDP
    parser.add_argument('--microbatch', type=int, default=0,
                        help='将每个进程上的 batch 按该大小切分做梯度累积，0 表示不启用')
    parser.add_argument('--ddp-find-unused', action='store_true',
                        help='启用 DDP 未用参数检测（稍慢）。若已正确冻结分支，可不加。')

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

    # 分布式 & 设备
    is_dist, rank, world, local_rank = setup_distributed()
    device = setup_device(local_rank)
    if is_main_process(rank):
        ngpu_vis = ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]) \
                   if torch.cuda.is_available() else 'CPU'
        print(f"DDP: {is_dist} | rank={rank}/{world} | device={device} | GPUs: {ngpu_vis}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = args.outdir if args.outdir is not None else args.log_dir
    log_dir = os.path.join(base_dir, f'adv_mamba_{timestamp}')
    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        # --- ADD: metrics accumulators (rank0 only) ---
        metrics = {
            'g_total': 0.0, 'g_spec': 0.0, 'g_adv': 0.0, 'g_channel': 0.0, 'g_phase': 0.0, 'g_n': 0,
            'd_loss': 0.0, 'd_real': 0.0, 'd_fake': 0.0, 'd_n': 0
        }

    else:
        writer = None  # 非主进程不写日志

    # -------------------------- 数据 --------------------------
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
    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=args.drop_last)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        drop_last=args.drop_last,
        collate_fn=collate_fn,
    )

    # -------------------------- 模型与恢复 --------------------------
    model = MambaEnhancedFarGan(in_features=F_used, cond_dim=32, subframe_size=40).to(device)

    if os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        msd = ckpt.get('model_state_dict', None)
        if msd is not None:
            model.load_state_dict(msd, strict=False)
            if is_main_process(rank):
                print(f'🔁 已载入预训练: {args.resume}')

    # ===== 冻结不会用到的分支（必须在 DDP 之前！）=====
    def freeze_by_keywords(module: torch.nn.Module, keywords):
        if not keywords:
            return []
        frozen = []
        for name, p in module.named_parameters():
            if any(kw in name for kw in keywords):
                p.requires_grad_(False)
                frozen.append(name)
        return frozen

    freeze_kw = []
    # 1) pitch 预测器当前实现未被调用，直接冻结
    freeze_kw += ["pitch_predictor"]
    # 2) 未启用 CSI → 冻结所有 CSI 门控 + 顶层自适应
    if not args.enable_csi:
        freeze_kw += ["cond_net.csi_gate", "subframe_net.csi_gates", "csi_adapt"]
        # 3) 未进行信道仿真 → 冻结 JSCC 编解码链
        freeze_kw += ["jscc_encoder", "jscc_decoder"]

    frozen_names = freeze_by_keywords(model, freeze_kw)
    if is_main_process(rank):
        print("[Freeze-before-DDP] keywords:", freeze_kw)
        print(f"[Freeze-before-DDP] frozen param count: {len(frozen_names)}")
        for n in frozen_names[:20]:
            print("  -", n)
        if len(frozen_names) > 20:
            print(f"  ... (+{len(frozen_names)-20} more)")

    # 可选：编译（放在冻结之后、DDP 之前）
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank):
            print('🔥 compile model (inductor)')
        model = torch.compile(model, mode='max-autotune')

    # 判别器（两个频域判别器）
    disc1 = osce_models.model_dict['fdmresdisc'](
        architecture='free', design='f_down',
        fft_sizes_16k=[2**n for n in range(6, 12)],
        freq_roi=[0, 7400], max_channels=args.disc1_max_ch, noise_gain=0.0
    ).to(device)
    disc2 = osce_models.model_dict['fdmresdisc'](
        architecture='free', design='f_down',
        fft_sizes_16k=[2**n for n in range(7, 11)],
        freq_roi=[0, 8000], max_channels=args.disc2_max_ch, noise_gain=0.1
    ).to(device)
    discriminators = [disc1, disc2]

    # 可选：channels_last（1D 上收益不一定显著）
    try:
        model = model.to(memory_format=torch.channels_last)
        for d in discriminators:
            d.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # DDP 包装（现在可将 find_unused 关闭；若调试可开启）
    if is_dist:
        find_unused = getattr(args, "ddp_find_unused", False)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=find_unused
        )
        discriminators = [
            nn.parallel.DistributedDataParallel(
                d, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False  # 判别器 G 步不会反传其梯度，这里保持 False
            ) for d in discriminators
        ]

    # -------------------------- 损失 & 优化器 --------------------------
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    loss_fn = MambaJSCCEnhancedLoss(
        spectral_weight=1.0,
        adversarial_weight=0.1,
        channel_weight=0.05 if args.enable_csi else 0.0,
        phase_weight=phase_weight,
        stft_sizes=args.stft_sizes
    )

    # 只把仍然可训练的参数交给生成器优化器
    optimizer_g = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr_g, betas=(0.8, 0.99), weight_decay=1e-4
    )
    optimizer_d = torch.optim.AdamW([p for d in discriminators for p in d.parameters()],
                                    lr=args.lr_d, betas=(0.8, 0.99))

    # AMP
    use_amp = args.mixed_precision and (device.type == 'cuda')
    try:
        from torch import amp as _amp
        autocast_ctx = lambda: _amp.autocast(device_type='cuda', enabled=use_amp)
        scaler = _amp.GradScaler('cuda') if use_amp and hasattr(_amp, 'GradScaler') else None
    except Exception:
        from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
        scaler = _GradScaler(enabled=use_amp)
        autocast_ctx = lambda: _autocast(enabled=use_amp)

    # -------------------------- 训练循环 --------------------------
    step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        if is_dist:
            dl.sampler.set_epoch(epoch)  # 保证各 rank shuffle 不同
        if is_main_process(rank):
            tepoch = tqdm(dl, desc=f'Adv Epoch {epoch+1}/{args.epochs}')
        else:
            tepoch = dl

        # train 模式
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.train()
        else:
            model.train()
        for d in discriminators:
            d.train()

        for bidx, (features, target) in enumerate(tepoch):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 随机CSI/信道（按整个 batch 生成）
            csi = None
            channel_noise = None
            if args.enable_csi:
                B = features.shape[0]
                csi = torch.empty(B, device=device, dtype=torch.float32).uniform_(args.snr_range[0], args.snr_range[1])
                if torch.rand(1, device=device).item() < args.channel_prob:
                    channel_noise = 'auto'

            # ======== 判别器用的假样本（no_grad，避免建图）========
            N_samples = min(target.shape[1], features.shape[1] * 160)
            with torch.no_grad():
                with autocast_ctx():
                    y_hat_for_d = model(
                        features,
                        csi=csi,
                        channel_noise=channel_noise,
                        target_length=N_samples,
                        parallel_train=args.parallel_train,
                        teacher_signal=target if args.parallel_train else None,
                    )
                    min_len = min(y_hat_for_d.shape[1], target.shape[1])
                    y_hat_for_d = y_hat_for_d[:, :min_len]
                    target_full = target[:, :min_len]

            # ======== 判别器更新（D step）========
            update_d = (args.adv_every <= 1) or (step % args.adv_every == 0)
            if update_d:
                for d in discriminators:
                    d.train()
                    for p in d.parameters():
                        p.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)

                from contextlib import ExitStack
                mb_pairs = list(iter_microbatches(target_full, y_hat_for_d, microbatch=args.microbatch))
                num_mb = len(mb_pairs)
                if num_mb == 0:
                    continue  # 保险：空微批直接跳过

                for i, (t_mb, y_mb) in enumerate(mb_pairs):
                    if is_dist and i < num_mb - 1:
                        with ExitStack() as stack:
                            for d in discriminators:
                                if isinstance(d, nn.parallel.DistributedDataParallel):
                                    stack.enter_context(d.no_sync())
                            with autocast_ctx():
                                d_loss_mb = torch.zeros([], device=device)
                                r_means, f_means = [], []
                                n_terms = 0  # <<< ADD: 正确的项数统计
                                for disc in discriminators:
                                    real_scores = disc(t_mb.unsqueeze(1).contiguous())
                                    fake_scores = disc(y_mb.unsqueeze(1).contiguous())
                                    for r_s, f_s in zip(real_scores, fake_scores):
                                        d_loss_mb = d_loss_mb + F.mse_loss(r_s[-1], torch.ones_like(r_s[-1]))
                                        d_loss_mb = d_loss_mb + F.mse_loss(f_s[-1], torch.zeros_like(f_s[-1]))
                                        n_terms += 1  # <<< ADD
                                    r_means.append(torch.stack([r[-1].mean() for r in real_scores]).mean())
                                    f_means.append(torch.stack([f[-1].mean() for f in fake_scores]).mean())
                                d_loss_mb = d_loss_mb / max(n_terms, 1)  # <<< FIX: 用累计项数归一化


                            # accumulate (rank0)  <<< stays the same but用 r_means/f_means
                            if is_main_process(rank):
                                with torch.no_grad():
                                    r_mean = torch.stack(r_means).mean().item()
                                    f_mean = torch.stack(f_means).mean().item()
                                    metrics['d_loss'] += float(d_loss_mb.detach().float().item())
                                    metrics['d_real'] += r_mean
                                    metrics['d_fake'] += f_mean
                                    metrics['d_n']    += 1

                            if scaler is not None:
                                scaler.scale(d_loss_mb).backward()
                            else:
                                d_loss_mb.backward()
                    else:
                        with autocast_ctx():
                            d_loss_mb = torch.zeros([], device=device)
                            r_means, f_means = [], []
                            n_terms = 0  # <<< ADD
                            for disc in discriminators:
                                real_scores = disc(t_mb.unsqueeze(1).contiguous())
                                fake_scores = disc(y_mb.unsqueeze(1).contiguous())
                                for r_s, f_s in zip(real_scores, fake_scores):
                                    d_loss_mb = d_loss_mb + F.mse_loss(r_s[-1], torch.ones_like(r_s[-1]))
                                    d_loss_mb = d_loss_mb + F.mse_loss(f_s[-1], torch.zeros_like(f_s[-1]))
                                    n_terms += 1  # <<< ADD
                                r_means.append(torch.stack([r[-1].mean() for r in real_scores]).mean())
                                f_means.append(torch.stack([f[-1].mean() for f in fake_scores]).mean())
                            d_loss_mb = d_loss_mb / max(n_terms, 1)  # <<< FIX


                        if is_main_process(rank):
                            with torch.no_grad():
                                r_mean = torch.stack(r_means).mean().item()
                                f_mean = torch.stack(f_means).mean().item()
                                metrics['d_loss'] += float(d_loss_mb.detach().float().item())
                                metrics['d_real'] += r_mean
                                metrics['d_fake'] += f_mean
                                metrics['d_n']    += 1

                        if scaler is not None:
                            scaler.scale(d_loss_mb).backward()
                        else:
                            d_loss_mb.backward()
                # AMP：先反缩放再裁剪再 step
                if scaler is not None:
                    scaler.unscale_(optimizer_d)
                for d in discriminators:
                    torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer_d)
                else:
                    optimizer_d.step()

            # ======== 生成器更新（G step）========
            for d in discriminators:
                d.eval()
                for p in d.parameters():
                    p.requires_grad_(False)

            optimizer_g.zero_grad(set_to_none=True)

            mb_triplets = list(iter_microbatches(features, target_full, microbatch=args.microbatch))
            if len(mb_triplets) == 0:
                continue

            offset = 0  # 新增：用来切分 csi
            for i, (f_mb, t_mb) in enumerate(mb_triplets):
                no_sync_ctx = None
                if is_dist and i < len(mb_triplets) - 1 and isinstance(model, nn.parallel.DistributedDataParallel):
                    no_sync_ctx = model.no_sync(); no_sync_ctx.__enter__()

                # === 新增：为当前微批切出 csi_mb，并确保始终传入 channel_noise ===
                bsz = t_mb.shape[0]
                csi_mb = csi[offset:offset+bsz] if csi is not None else None
                offset += bsz

                with autocast_ctx():
                    y_mb = model(
                        f_mb,
                        csi=csi_mb,                 # 原先是 None，这里改为微批对应的 csi
                        channel_noise=channel_noise,# 原先是 None，这里统一传 batch 的设定（'auto' 或 None）
                        target_length=t_mb.shape[1],
                        parallel_train=args.parallel_train,
                        teacher_signal=t_mb if args.parallel_train else None,
                    )

                # 需要对抗项时，跑判别器前向用于 adv/feat-match
                if update_d:
                    with autocast_ctx():
                        gen_scores_all = [disc(y_mb.unsqueeze(1).contiguous()) for disc in discriminators]
                else:
                    gen_scores_all = None

                with autocast_ctx():
                    g_losses = loss_fn(
                        pred=y_mb,
                        target=t_mb,
                        csi=csi_mb,             # 可选：若你的 loss 里要用 csi 自适应项，这里也传进去
                        disc_real=None,
                        disc_fake=gen_scores_all,
                    )
                    g_total = g_losses['total']
                    # --- ADD: accumulate G metrics (rank0 only) ---
                    if is_main_process(rank):
                        metrics['g_total']   += float(g_losses['total'].detach().float().item())
                        def _to_float(x):
                            if isinstance(x, torch.Tensor):
                                return float(x.detach().float().item())
                            return float(x)
                        metrics['g_spec']    += _to_float(g_losses.get('spectral',    0.0))
                        metrics['g_adv']     += _to_float(g_losses.get('adversarial', 0.0))
                        metrics['g_channel'] += _to_float(g_losses.get('channel',     0.0))
                        metrics['g_phase']   += _to_float(g_losses.get('phase',       0.0))
                        metrics['g_n']       += 1

                if scaler is not None:
                    scaler.scale(g_total).backward()
                else:
                    g_total.backward()

                if no_sync_ctx is not None:
                    no_sync_ctx.__exit__(None, None, None)

            # AMP 下先反缩放再裁剪，再 step/update（保持不变）
            if scaler is not None:
                scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer_g)
                scaler.update()
            else:
                optimizer_g.step()


            step += 1

            # 日志（仅主进程）
            if is_main_process(rank) and (step % args.log_interval == 0):
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-6)
                writer.add_scalar('Perf/steps_per_sec', sps, step)
                # --- ADD: write averaged G/D metrics ---
                if metrics['g_n'] > 0:
                    writer.add_scalar('G/loss_total',    metrics['g_total']/metrics['g_n'],   step)
                    writer.add_scalar('G/loss_spectral', metrics['g_spec']/metrics['g_n'],    step)
                    writer.add_scalar('G/loss_adv',      metrics['g_adv']/metrics['g_n'],     step)
                    writer.add_scalar('G/loss_channel',  metrics['g_channel']/metrics['g_n'], step)
                    writer.add_scalar('G/loss_phase',    metrics['g_phase']/metrics['g_n'],   step)
                if metrics['d_n'] > 0:
                    writer.add_scalar('D/loss',            metrics['d_loss']/metrics['d_n'], step)
                    writer.add_scalar('D/real_logit_mean', metrics['d_real']/metrics['d_n'], step)
                    writer.add_scalar('D/fake_logit_mean', metrics['d_fake']/metrics['d_n'], step)

                # reset accumulators
                metrics.update({k: 0.0 for k in metrics})
                metrics['g_n'] = 0; metrics['d_n'] = 0
            if args.max_steps_per_epoch and (bidx + 1) >= args.max_steps_per_epoch:
                break

        # 同步，确保各 rank 结束本 epoch
        if is_dist:
            torch.distributed.barrier()
        # --- ADD: tiny validation on rank0 (one small batch) ---
        if is_main_process(rank):
            model_eval = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            model_eval.eval()
            with torch.no_grad():
                try:
                    vfeat, vtarget = next(iter(dl))   # 直接用训练 loader 抽样观察（可替换为独立 val_loader）
                    vfeat   = vfeat[:2].to(device)    # 只取两条，极小开销
                    vtarget = vtarget[:2].to(device)
                    yv = model_eval(
                        vfeat, csi=None, channel_noise=None,
                        target_length=min(vtarget.shape[1], vfeat.shape[1]*160),
                        parallel_train=False
                    )
                    yv = yv[:, :vtarget.shape[1]]

                    # 先尝试用第一个 STFT 分量作为 proxy；若没有该属性则退化为 L1
                    val_scalar_written = False
                    try:
                        v_stft = loss_fn.stft_losses[0](yv, vtarget).detach().float().item()
                        writer.add_scalar('Val/stftloss_1024', v_stft, step)
                        val_scalar_written = True
                    except Exception:
                        pass
                    if not val_scalar_written:
                        v_l1 = F.l1_loss(yv, vtarget).detach().float().item()
                        writer.add_scalar('Val/l1_wave', v_l1, step)

                    # 写一段音频，方便随时听效果
                    try:
                        writer.add_audio('Val/pred',   yv[0].detach().cpu(), step, sample_rate=16000)
                        writer.add_audio('Val/target', vtarget[0].detach().cpu(), step, sample_rate=16000)
                    except Exception:
                        pass
                except StopIteration:
                    pass
            model_eval.train()

        # 保存（仅主进程）
        if is_main_process(rank) and ((epoch + 1) % args.save_every == 0):
            save_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            ckpt = {
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'args': vars(args),
            }
            ckpt_path = os.path.join(log_dir, f'adv_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f'💾 保存对抗检查点: {ckpt_path}')

    if is_main_process(rank) and writer is not None:
        writer.close()

    # 优雅退出分布式
    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
