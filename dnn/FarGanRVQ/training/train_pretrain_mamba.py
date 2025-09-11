#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MambaEnhancedFarGan 预训练脚本（无对抗，DDP）
- 仅重建类损失（多分辨率STFT/相位/信道自适应）
- 支持 DDP / AMP / torch.compile / Teacher Forcing
- 记录分项损失、波形统计、轻量验证、best.pth（EMA权重）
- 关键：在 DDP 之前“预冻结未使用分支”（pitch/CSI/JSCC），避免 unused params 报错
"""

import argparse, os, sys, time, math
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------- 路径 ----------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'dnn/torch/fargan'))

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.mamba_enhanced_fargan import (
    MambaEnhancedFarGan, MambaJSCCEnhancedLoss
)

# ---------- 预处理：去直流 + RMS归一 + anti-clip ----------
def preprocess_for_loss(pred: torch.Tensor, target: torch.Tensor, ref_rms: float = 0.1):
    # 去直流
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    # RMS 归一
    eps = 1e-8
    p_rms = torch.sqrt((pred ** 2).mean(dim=1, keepdim=True) + eps)
    t_rms = torch.sqrt((target ** 2).mean(dim=1, keepdim=True) + eps)
    pred = pred * (ref_rms / p_rms)
    target = target * (ref_rms / t_rms)
    # anti-clip（软限制）
    pred = 0.999 * torch.tanh(pred / 0.999)
    return pred, target

# ---------- EMA ----------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

# ---------- Warmup + Cosine ----------
def build_warmup_cosine(optimizer, total_steps: int, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------- DDP ----------
def ddp_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    return False, 0, 1, 0

def setup_distributed():
    is_dist, rank, world, local = ddp_env()
    if is_dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return is_dist, rank, world, local

def is_main_process(rank): return rank == 0

# ---------- collate ----------
def collate_fn(batch):
    feats, data = [], []
    for features, periods, waveform, lpc in batch:
        feats.append(features); data.append(waveform)
    feats = torch.from_numpy(np.array(feats)).float()
    data  = torch.from_numpy(np.array(data)).float()
    return feats, data

# ---------- 预冻结工具 ----------
def freeze_by_keywords(module: torch.nn.Module, keywords):
    frozen = []
    for name, p in module.named_parameters():
        if any(kw in name for kw in keywords):
            p.requires_grad_(False)
            frozen.append(name)
    return frozen

def main():
    ap = argparse.ArgumentParser(description='MambaEnhancedFarGan 预训练（无对抗, DDP）')
    ap.add_argument('--unfreeze-pitch', action='store_true', help='polish 阶段解冻 pitch_predictor')
    ap.add_argument('--features', type=str, required=True)
    ap.add_argument('--pcm', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--seq-len', type=int, default=60)
    ap.add_argument('--log-dir', type=str, default='dnn/FarGanRVQ/tensorboard_logs')
    ap.add_argument('--outdir', type=str, default=None)
    ap.add_argument('--log-interval', type=int, default=50)
    ap.add_argument('--save-every', type=int, default=20)
    # 性能/功能
    ap.add_argument('--parallel-train', action='store_true', help='Teacher Forcing（并行子帧）')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--prefetch-factor', type=int, default=6)
    ap.add_argument('--drop-last', action='store_true', default=True)
    ap.add_argument('--max-steps-per-epoch', type=int, default=0)
    ap.add_argument('--channels-last', action='store_true')
    # 损失/信道
    ap.add_argument('--enable-csi', action='store_true')
    ap.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20])
    ap.add_argument('--channel-prob', type=float, default=0.5)
    ap.add_argument('--stft-sizes', type=int, nargs='+', default=[512, 1024])
    ap.add_argument('--disable-phase-loss', action='store_true')
    # DDP 相关
    ap.add_argument('--ddp-find-unused', action='store_true', help='调试时可开；性能较差')
    ap.add_argument('--no-freeze-unused', dest='freeze_unused', action='store_false', help='关闭预冻结（默认开启）')
    ap.set_defaults(freeze_unused=True)

    args = ap.parse_args()

    # ---------- 分布式/设备 ----------
    is_dist, rank, world, local_rank = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if is_dist else 'cuda')
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')

    torch.manual_seed(1337 + rank); np.random.seed(1337 + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1337 + rank)

    # ---------- 日志 ----------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = args.outdir if args.outdir is not None else args.log_dir
    log_dir   = os.path.join(base_dir, f'pretrain_mamba_{timestamp}')
    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # ---------- 数据 ----------
    frame_size = 160
    seq_len    = args.seq_len
    F_used     = 20
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
        sampler = None; shuffle = True

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

    # ---------- 模型 ----------
    model = MambaEnhancedFarGan(in_features=F_used, cond_dim=32, subframe_size=40).to(device)

    # 预冻结未用分支（必须在 DDP 之前）
    if args.freeze_unused:
        freeze_kw = []
        if not args.unfreeze_pitch:
            freeze_kw += ["pitch_predictor"]
        if not args.enable_csi:
            freeze_kw += ["cond_net.csi_gate", "subframe_net.csi_gates", "csi_adapt",
                          "jscc_encoder", "jscc_decoder"]
        frozen = freeze_by_keywords(model, freeze_kw)
        if is_main_process(rank):
            print(f"[Freeze-before-DDP] frozen params: {len(frozen)}")
            for n in frozen[:20]:
                print("  -", n)
            if len(frozen) > 20:
                print(f"  ... (+{len(frozen)-20} more)")

    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if args.compile and hasattr(torch, 'compile'):
        if is_main_process(rank): print('🔥 compile model (inductor)')
        model = torch.compile(model, mode='max-autotune')

    # DDP 包装
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused
        )

    # ---------- 损失 ----------
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    loss_fn = MambaJSCCEnhancedLoss(
        spectral_weight=1.0,
        adversarial_weight=0.0,
        channel_weight=0.05 if args.enable_csi else 0.0,
        phase_weight=phase_weight,
        stft_sizes=args.stft_sizes,
    )

    # ---- 优化器（只初始化一次）----
    base_model = (model.module if is_dist else model)
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)

    # ---- scheduler（warmup + cosine）----
    steps_per_epoch = len(dl) if args.max_steps_per_epoch == 0 else min(len(dl), args.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = build_warmup_cosine(optimizer, total_steps, warmup_ratio=0.05, min_lr_ratio=0.1)

    # ---- EMA ----
    ema = EMA(base_model, decay=0.999)

    # ---- AMP ----
    use_amp = args.mixed_precision and (device.type == 'cuda')
    try:
        from torch import amp as _amp
        autocast_ctx = lambda: _amp.autocast(device_type='cuda', enabled=use_amp)
        scaler = _amp.GradScaler('cuda') if use_amp and hasattr(_amp, 'GradScaler') else None
    except Exception:
        from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
        autocast_ctx = lambda: _autocast(enabled=use_amp)
        scaler = _GradScaler(enabled=use_amp)

    # ---- hparams ----
    if is_main_process(rank) and writer is not None:
        total_params = sum(p.numel() for p in (model.module if is_dist else model).parameters())
        writer.add_hparams({
            'batch_size(per-rank)': args.batch_size,
            'lr': args.lr, 'seq_len': seq_len,
            'model_params': total_params,
            'parallel_train': args.parallel_train,
            'stft_sizes': str(args.stft_sizes),
            'freeze_unused': args.freeze_unused,
            'ddp_find_unused': args.ddp_find_unused,
        }, {})

    # ---------- 训练 ----------
    best = float('inf'); step = 0; start_time = time.time()
    for epoch in range(args.epochs):
        if is_dist and sampler is not None:
            sampler.set_epoch(epoch)

        (model.module if is_dist else model).train()
        tepoch = tqdm(dl, desc=f'Pretrain Epoch {epoch+1}/{args.epochs}', disable=not is_main_process(rank))

        for bidx, (features, target) in enumerate(tepoch):
            features = features.to(device, non_blocking=True)
            target  = target.to(device,  non_blocking=True)

            N_samples = min(target.shape[1], features.shape[1] * 160)
            with autocast_ctx():
                y_hat = model(
                    features,
                    csi=None, channel_noise=None,
                    target_length=N_samples,
                    parallel_train=args.parallel_train,
                    teacher_signal=target if args.parallel_train else None,
                )
                # 对齐
                min_len = min(y_hat.shape[1], target.shape[1])
                y_hat = y_hat[:, :min_len]
                target_ = target[:, :min_len]

                # 去直流 + RMS 归一 + anti-clip（仅用于 loss）
                y_hat_p, target_p = preprocess_for_loss(y_hat, target_, ref_rms=0.1)

                # 损失
                losses = loss_fn(pred=y_hat_p, target=target_p, csi=None, disc_real=None, disc_fake=None)
                loss = losses['total']

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_((model.module if is_dist else model).parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_((model.module if is_dist else model).parameters(), 1.0)
                optimizer.step()

            # EMA & scheduler
            ema.update(base_model)
            scheduler.step()

            step += 1
            if is_main_process(rank):
                tepoch.set_postfix({'L': f'{float(loss.item()):.4f}'})
                if writer is not None:
                    # 分项损失
                    if 'spectral' in losses: writer.add_scalar('Loss/spec', float(losses['spectral']), step)
                    if 'phase'    in losses: writer.add_scalar('Loss/phase', float(losses['phase']), step)
                    if 'channel'  in losses: writer.add_scalar('Loss/channel', float(losses['channel']), step)
                    # 波形统计（原始 y_hat）
                    with torch.no_grad():
                        p = y_hat.detach(); t = target_
                        def _rms(x): return float((x**2).mean().sqrt().item())
                        def _dc(x):  return float(x.mean().item())
                        clip_rate = float(((p<-0.999)|(p>0.999)).float().mean().item())
                        writer.add_scalar('Wave/pred_rms', _rms(p), step)
                        writer.add_scalar('Wave/tgt_rms',  _rms(t), step)
                        writer.add_scalar('Wave/pred_dc',  _dc(p),  step)
                        writer.add_scalar('Wave/clip_rate', clip_rate, step)
                    # 学习率 & 性能
                    if step % args.log_interval == 0:
                        elapsed = time.time() - start_time
                        sps = step / max(elapsed, 1e-6)
                        writer.add_scalar('Loss/pretrain_total', float(loss.item()), step)
                        writer.add_scalar('Perf/steps_per_sec', sps, step)
                        writer.add_scalar('Opt/lr', optimizer.param_groups[0]['lr'], step)

            if args.max_steps_per_epoch and (bidx + 1) >= args.max_steps_per_epoch:
                break

        # 轻量验证（rank0）
        if is_main_process(rank):
            model_eval = model.module if is_dist else model
            model_eval.eval()
            val_metric = float('inf'); metric_name = 'Val/l1_wave'

            # 用 EMA 权重验证
            ema.apply_shadow(model_eval)
            with torch.no_grad():
                try:
                    vfeat, vtarget = next(iter(dl))
                except StopIteration:
                    vfeat, vtarget = None, None
                if vfeat is not None:
                    vfeat = vfeat[:2].to(device); vtarget = vtarget[:2].to(device)
                    fixed_tlen = args.seq_len * 160
                    with autocast_ctx():
                        yv = model_eval(vfeat, csi=None, channel_noise=None,
                                        target_length=fixed_tlen, parallel_train=False)
                    yv = yv[:, :min(vtarget.shape[1], yv.shape[1])]
                    vt = vtarget[:, :yv.shape[1]]

                    # 粗糙互相关对齐
                    def best_lag_1d(x, y, max_lag=800):
                        best_lag, best_c = 0, -1e9
                        for L in range(-max_lag, max_lag+1):
                            if L >= 0: c = F.cosine_similarity(x[L:], y[:y.numel()-L], dim=0)
                            else:
                                L2 = -L; c = F.cosine_similarity(x[:x.numel()-L2], y[L2:], dim=0)
                            v = float(c)
                            if v > best_c: best_c, best_lag = v, L
                        return best_lag, best_c
                    lag, corr = best_lag_1d(yv[0].contiguous().view(-1), vt[0].contiguous().view(-1), 800)
                    def align(x, y, L):
                        if L>0:  return x[:, L:], y[:, :y.shape[1]-L]
                        if L<0:  return x[:, :x.shape[1]+L], y[:, -L:]
                        return x, y
                    yv_a, vt_a = align(yv, vt, lag)
                    L_ = min(yv_a.shape[1], vt_a.shape[1]); yv_a, vt_a = yv_a[:, :L_], vt_a[:, :L_]

                    try:
                        if hasattr(loss_fn, 'stft_losses') and len(getattr(loss_fn, 'stft_losses', []))>0:
                            v_stft = loss_fn.stft_losses[0](yv_a, vt_a).detach().float().item()
                            val_metric = float(v_stft); metric_name = 'Val/stftloss_win'
                        else:
                            val_metric = float(F.l1_loss(yv_a, vt_a).detach().float().item())
                            metric_name = 'Val/l1_wave'
                    except Exception:
                        val_metric = float(F.l1_loss(yv_a, vt_a).detach().float().item())
                        metric_name = 'Val/l1_wave'

                    if writer is not None:
                        writer.add_scalar(metric_name, val_metric, step)
                        writer.add_scalar('Val/lag_samples', float(lag), step)
                        writer.add_scalar('Val/corr', float(corr), step)
                        try:
                            # 安全写音频：去除 NaN/Inf 并限幅到 [-1, 1]
                            p_audio = yv_a[0].detach().cpu().float()
                            t_audio = vt_a[0].detach().cpu().float()
                            p_audio = torch.nan_to_num(p_audio, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                            t_audio = torch.nan_to_num(t_audio, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                            writer.add_audio('Val/pred_aligned',   p_audio, step, sample_rate=16000)
                            writer.add_audio('Val/target_aligned', t_audio, step, sample_rate=16000)
                        except Exception:
                            pass
            # 恢复即时权重
            ema.restore(model_eval)

            # best & 保存（EMA 权重）
            if not np.isfinite(val_metric): val_metric = float('inf')
            if val_metric + 1e-6 < best:
                best = val_metric
                save_model = model.module if is_dist else model
                ema.apply_shadow(save_model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                    'best_metric': best,
                    'metric_name': metric_name,
                    'ema': True,
                    'ema_shadow': {k: v.cpu() for k, v in ema.shadow.items()},
                    'ema_decay': ema.decay,
                }, os.path.join(log_dir, 'best.pth'))
                ema.restore(save_model)
                print(f'🏅 验证提升：best={best:.6f}（{metric_name}），已保存 best.pth')

            # 周期性保存（即时 + EMA）
            if (epoch + 1) % args.save_every == 0:
                save_model = model.module if is_dist else model

                # 1) 即时权重
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                    'ema': False,
                }
                inst_path = os.path.join(log_dir, f'pretrain_epoch_{epoch+1}.pth')
                torch.save(ckpt, inst_path)
                print(f'💾 保存预训练检查点(instant): {inst_path}')

                # 2) EMA 权重
                try:
                    ema.apply_shadow(save_model)
                    ckpt_ema = {
                        'epoch': epoch,
                        'model_state_dict': save_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': vars(args),
                        'ema': True,
                        'ema_shadow': {k: v.cpu() for k, v in ema.shadow.items()},
                        'ema_decay': ema.decay,
                    }
                    ema_path = os.path.join(log_dir, f'pretrain_epoch_{epoch+1}_ema.pth')
                    torch.save(ckpt_ema, ema_path)
                    print(f'💾 保存预训练检查点(EMA): {ema_path}')
                finally:
                    ema.restore(save_model)

        if is_dist:
            torch.distributed.barrier()

    if is_main_process(rank) and writer is not None:
        writer.close()
    if is_dist:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
