#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MambaEnhancedFarGan 抛光训练脚本（非对抗版）
- 仅使用重建类损失（Multi-Res STFT / 可选 Phase / 可选 CSI）
- 不构建判别器、不计算对抗项，运行/显存更轻
- 冻结抛光阶段用不到的分支（pitch/JSCC/CSI 门控），避免 DDP 未用参数错误
- 支持单机多卡 DDP（torchrun）、AMP、微批累积、按步保存、早停
- 轻量化验证：STFT 验证指标（可选 log-mel）
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
from contextlib import nullcontext

# --------- 路径 ---------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
fargan_path = os.path.join(project_root, 'dnn/torch/fargan')
sys.path.insert(0, fargan_path)

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.mamba_enhanced_fargan import (
    MambaEnhancedFarGan, MambaJSCCEnhancedLoss
)


# --------- 分布式/设备 ---------
def ddp_env():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world, local
    return False, 0, 1, 0

def setup_distributed():
    is_dist, rank, world, local = ddp_env()
    if is_dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return is_dist, rank, world, local

def is_main(rank): return rank == 0

def setup_device(local_rank: int):
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(dev)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
    else:
        dev = torch.device('cpu')
    return dev


# --------- 小工具 ---------
def iter_microbatches(*tensors, microbatch: int):
    """按 batch 维切微批；microbatch<=0 时返回整体"""
    if microbatch is None or microbatch <= 0:
        yield tensors; return
    B = tensors[0].shape[0]
    for i in range(0, B, microbatch):
        sl = slice(i, min(i + microbatch, B))
        yield tuple(t[sl] if t is not None else None for t in tensors)

def save_ckpt(save_model, optimizer, epoch, step, log_dir, name, extra: dict = None):
    ckpt = {
        'epoch': epoch, 'step': step,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'saved_at': datetime.now().isoformat(timespec='seconds'),
    }
    if extra: ckpt.update(extra)
    path = os.path.join(log_dir, name)
    torch.save(ckpt, path)
    print(f'💾 已保存: {path}')


# --------- 主逻辑 ---------
def main():
    parser = argparse.ArgumentParser(description='MambaEnhancedFarGan 抛光训练（非对抗）')

    # 数据/模型
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--pcm', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True, help='预训练或已有检查点(.pth)')
    parser.add_argument('--in-features', type=int, default=20)
    parser.add_argument('--seq-len', type=int, default=48, help='抛光可用 40~60 之间')
    parser.add_argument(
        '--unfreeze-pitch',
        action='store_true',
        help='在 polish 阶段解冻 pitch_predictor（其余可不必要分支保持冻结）'
    )

    # 训练超参
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps-per-epoch', type=int, default=1200)
    parser.add_argument('--microbatch', type=int, default=0, help='>0 时启用梯度累积')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--simple-loss', action='store_true',
                        help='用 waveform L1 代替 STFT/phase（仅用于性能诊断）')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default','reduce-overhead','max-autotune'])
    parser.add_argument('--parallel-train', action='store_true',
                    help='抛光阶段也走并行/teacher-forcing 路径')

    # 损失/CSI
    parser.add_argument('--stft-sizes', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--disable-phase-loss', action='store_true')
    parser.add_argument('--enable-csi', action='store_true', help='默认关闭；抛光阶段建议关闭更省算')
    parser.add_argument('--snr-range', type=float, nargs=2, default=[-10, 20])
    parser.add_argument('--channel-prob', type=float, default=0.5)

    # Loader 性能
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--prefetch-factor', type=int, default=8)
    parser.add_argument('--drop-last', action='store_true', default=True)

    # 日志/保存/早停
    parser.add_argument('--outdir', type=str, default='dnn/FarGanRVQ/checkpoints/polish_runs')
    parser.add_argument('--tag', type=str, default='polish')
    parser.add_argument('--log-interval', type=int, default=400)
    parser.add_argument('--save-latest-every', type=int, default=500)
    parser.add_argument('--save-every-steps', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=6, help='验证指标连续 N 次无提升则早停')

    # 验证细节
    parser.add_argument('--val-audio-every', type=int, default=1000, help='每 N 步写一次音频到 TB；0=关闭')
    parser.add_argument('--val-mel', action='store_true', help='额外计算 log-mel L1（稍慢）')

    # 诊断（可选）
    parser.add_argument('--diag', action='store_true',
                        help='记录每步耗时(dt_load/fwd/bwd/opt/step)到TB并打印')

    args = parser.parse_args()

    # 分布式 & 设备
    is_dist, rank, world, local_rank = setup_distributed()
    device = setup_device(local_rank)
    if is_main(rank):
        ngpu = ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]) \
               if torch.cuda.is_available() else 'CPU'
        print(f"DDP={is_dist} | rank={rank}/{world} | device={device} | GPUs: {ngpu}")
        if args.diag:
            print("⚙️  DIAG 已开启：Perf/dt_* 将写入 TensorBoard")

    # 日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.outdir, f'{args.tag}_{timestamp}')
    if is_main(rank):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # 数据集
    frame_size = 160  # 10ms @16k
    ds = FARGANDataset(
        feature_file=args.features,
        signal_file=args.pcm,
        frame_size=frame_size,
        sequence_length=args.seq_len,
        lookahead=1,
        nb_used_features=args.in_features,
        nb_features=36,
    )
    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=args.drop_last)
        shuffle = False
    else:
        sampler = None; shuffle = True

    def _collate(batch):
        # batch item: (features, periods, waveform, lpc)
        feats = torch.from_numpy(np.array([b[0] for b in batch])).float()
        waves = torch.from_numpy(np.array([b[2] for b in batch])).float()
        return feats, waves

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
        collate_fn=_collate
    )

    # 模型
    model = MambaEnhancedFarGan(in_features=args.in_features, cond_dim=32, subframe_size=40).to(device)

    # 加载预训练/已有权重
    if os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location='cpu', weights_only=True)
        msd = sd.get('model_state_dict', sd if isinstance(sd, dict) else None)
        if msd: model.load_state_dict(msd, strict=False)
        if is_main(rank): print(f'🔁 已从 {args.resume} 载入权重')

    # ===== 冻结抛光阶段用不到的分支（在 DDP 之前！）=====
    def freeze_by_keywords(module: torch.nn.Module, keywords):
        if not keywords: return []
        frozen = []
        for name, p in module.named_parameters():
            if any(kw in name for kw in keywords):
                p.requires_grad_(False)
                frozen.append(name)
        return frozen

    freeze_kw = []

    # 没开 CSI/不做信道仿真：这些分支在 polish 阶段一般不需要，统一冻结更稳
    if not getattr(args, 'enable_csi', False):
        freeze_kw += [
            "cond_net.csi_gate",
            "subframe_net.csi_gates",
            "csi_adapt",
            "jscc_encoder",
            "jscc_decoder",
        ]

    # pitch 分支：只有当没有 --unfreeze-pitch 时才冻结
    if not getattr(args, 'unfreeze_pitch', False):
        freeze_kw += ["pitch_predictor"]

    frozen_names = freeze_by_keywords(model, freeze_kw)

    if is_main(rank):
        print(f"[Freeze-before-DDP] frozen param count: {len(frozen_names)}")
        for n in frozen_names[:20]:
            print("  -", n)
        if len(frozen_names) > 20:
            print(f"  ... (+{len(frozen_names)-20} more)")

    # 可选编译（放在冻结之后、DDP 之前）
    if args.compile and hasattr(torch, 'compile'):
        if is_main(rank): print(f'🔥 torch.compile (inductor, mode={args.compile_mode})')
        model = torch.compile(model, mode=args.compile_mode)

    # --- DDP 包装 ---
    if is_dist:
        find_unused = getattr(args, "ddp_find_unused", False)
        if getattr(args, "unfreeze_pitch", False):
            find_unused = True

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,   # ✅ 关键
            static_graph=False                    # 保险：图有分支/形状变化时更稳
        )

    # 损失（无对抗）
    phase_weight = 0.0 if args.disable_phase_loss else 0.1
    if not args.simple_loss:
        loss_fn = MambaJSCCEnhancedLoss(
            spectral_weight=1.0,
            adversarial_weight=0.0,                       # 关闭对抗
            channel_weight=0.05 if args.enable_csi else 0.0,
            phase_weight=phase_weight,
            stft_sizes=args.stft_sizes,
        )
    else:
        loss_fn = None  # 训练时用简单 L1 波形损失

    # 优化器（优先 fused=True，失败回退）
    trainable_params = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).parameters()
    opt_params = (p for p in trainable_params if p.requires_grad)
    try:
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4, fused=True)
        if is_main(rank): print("Optimizer: AdamW(fused=True)")
    except TypeError:
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-4)
        if is_main(rank): print("Optimizer: AdamW(fused=False fallback)")

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

    # 训练状态
    step = 0
    best_metric = float('inf')
    no_improve = 0
    start_time = time.time()
    first_step_t0 = None
    prev_step_end = start_time  # for dt_load (diag)

    # 训练循环
    for epoch in range(args.epochs):
        if is_dist and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
        tepoch = tqdm(dl, desc=f'Polish Epoch {epoch+1}/{args.epochs}') if is_main(rank) else dl

        # train
        (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).train()

        for bidx, (features, target) in enumerate(tepoch):
            t0_step = time.time()
            dt_load = t0_step - prev_step_end  # 取数/拷贝耗时（粗粒度）

            if is_main(rank) and step == 0:
                first_step_t0 = time.time()

            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # CSI（抛光阶段默认不建议开启）
            csi = None
            channel_noise = None
            if args.enable_csi:
                B = features.shape[0]
                csi = torch.empty(B, device=device, dtype=torch.float32).uniform_(args.snr_range[0], args.snr_range[1])
                if torch.rand((), device=device).item() < args.channel_prob:
                    channel_noise = 'auto'

            # ---- 前向；微批累积 ----
            optimizer.zero_grad(set_to_none=True)
            mb_pairs = list(iter_microbatches(features, target, microbatch=args.microbatch))
            if len(mb_pairs) == 0:
                continue
            num_mbs = max(1, len(mb_pairs))  # ✅ 微批均值分母
            fixed_tlen = args.seq_len * 160  # ✅ 固定 target_length，稳定计算图与吞吐

            dt_fwd = 0.0; dt_bwd = 0.0; dt_opt = 0.0  # diag

            offset = 0  # 为 csi 的微批切片准备
            for i, (f_mb, t_mb) in enumerate(mb_pairs):
                # 只有最后一个 microbatch 同步梯度，其余用 no_sync() 加速
                if is_dist and isinstance(model, nn.parallel.DistributedDataParallel) and i < len(mb_pairs) - 1:
                    sync_ctx = model.no_sync()
                else:
                    sync_ctx = nullcontext()

                with sync_ctx:
                    bsz = t_mb.shape[0]
                    csi_mb = csi[offset:offset+bsz] if (args.enable_csi and csi is not None) else None
                    offset += bsz

                    with autocast_ctx():
                        t_fwd_s = time.time()
                        y_mb = model(
                            f_mb,
                            csi=csi_mb,
                            channel_noise=channel_noise,
                            target_length=fixed_tlen,             # 固定长度，稳定图
                            parallel_train=args.parallel_train,
                            teacher_signal=t_mb if args.parallel_train else None,
                        )
                        min_len = min(y_mb.shape[1], t_mb.shape[1])
                        y_mb = y_mb[:, :min_len]
                        t_mb = t_mb[:, :min_len]

                        if loss_fn is None:
                            loss = F.l1_loss(y_mb, t_mb) / num_mbs
                        else:
                            losses = loss_fn(pred=y_mb, target=t_mb, csi=csi_mb)
                            loss = losses['total'] / num_mbs
                        dt_fwd += (time.time() - t_fwd_s)

                    # 只 backward 一次（不要重复！）
                    t_bwd_s = time.time()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    dt_bwd += (time.time() - t_bwd_s)

            # ---- 微批结束：AMP 下先反缩放，再裁剪、再 step ----
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).parameters(),
                max_norm=1.0
            )

            t_opt_s = time.time()
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            dt_opt += (time.time() - t_opt_s)


            # 记录
            step += 1
            if is_main(rank) and step % args.log_interval == 0:
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-6)
                writer.add_scalar('Perf/steps_per_sec', sps, step)
                writer.add_scalar('G/loss_total', float(loss.detach().float().item()), step)
                if args.diag:
                    dt_step = time.time() - t0_step
                    writer.add_scalar('Perf/dt_load_s', dt_load, step)
                    writer.add_scalar('Perf/dt_fwd_s', dt_fwd, step)
                    writer.add_scalar('Perf/dt_bwd_s', dt_bwd, step)
                    writer.add_scalar('Perf/dt_opt_s', dt_opt, step)
                    writer.add_scalar('Perf/dt_step_s', dt_step, step)
                    print(f"[rank0 step {step}] load={dt_load:.3f}s, fwd={dt_fwd:.3f}s, bwd={dt_bwd:.3f}s, opt={dt_opt:.3f}s, step={dt_step:.3f}s")

            if is_main(rank) and step == 1 and first_step_t0 is not None:
                print(f"[冷启动] first_step={time.time()-first_step_t0:.3f}s")

            # 保存
            if is_main(rank):
                save_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                if args.save_latest_every > 0 and step % args.save_latest_every == 0:
                    save_ckpt(save_model, optimizer, epoch, step, log_dir, 'latest.pth')
                if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                    save_ckpt(save_model, optimizer, epoch, step, log_dir, f'step_{step}.pth')

            prev_step_end = time.time()

            # 中途截断
            if args.max_steps_per_epoch and (bidx + 1) >= args.max_steps_per_epoch:
                break

        # ===== 验证（rank0）=====
        if is_dist:
            torch.distributed.barrier()
        if is_main(rank):
            model_eval = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            model_eval.eval()
            with torch.no_grad():
                try:
                    vfeat, vtarget = next(iter(dl))
                except StopIteration:
                    pass
                else:
                    vfeat = vfeat[:2].to(device); vtarget = vtarget[:2].to(device)
                    fixed_tlen = args.seq_len * 160
                    with autocast_ctx():
                        tlen = min(vtarget.shape[1], fixed_tlen)
                        yv = model_eval(
                            vfeat, csi=None, channel_noise=None,
                            target_length=tlen,
                            parallel_train=False
                        )[:, :vtarget.shape[1]]
                        L = min(yv.shape[1], vtarget.shape[1])
                        yv = yv[:, :L].contiguous()
                        vtarget = vtarget[:, :L].contiguous()
                        # 主指标：第一个 STFT 验证（越小越好）
                        metric_name = None
                        try:
                            if loss_fn is not None and hasattr(loss_fn, 'stft_losses'):
                                v_stft = loss_fn.stft_losses[0](yv, vtarget).detach().float().item()
                                cur_metric = float(v_stft)
                                metric_name = f'Val/stftloss_{getattr(loss_fn.stft_losses[0], "win_length", "win")}'
                            else:
                                cur_metric = float(F.l1_loss(yv, vtarget).detach().float().item())
                                metric_name = 'Val/l1_wave'
                            writer.add_scalar(metric_name, cur_metric, step)
                        except Exception:
                            cur_metric = float(F.l1_loss(yv, vtarget).detach().float().item())
                            metric_name = 'Val/l1_wave'
                            writer.add_scalar(metric_name, cur_metric, step)

                        writer.add_scalar('Val/earlystop_metric', cur_metric, step)

                        # best + 早停
                        if cur_metric + 1e-6 < best_metric:
                            best_metric = cur_metric; no_improve = 0
                            save_ckpt(model_eval, optimizer, epoch, step, log_dir, 'best.pth',
                                      extra={'best_metric': best_metric, 'metric_name': metric_name})
                            print(f"🏅 验证提升：best={best_metric:.6f}（{metric_name}），已保存 best.pth")
                        else:
                            no_improve += 1
                            if args.patience > 0 and no_improve >= args.patience:
                                print(f"⏹️ 早停：连续 {no_improve} 次无提升（metric={cur_metric:.6f}, best={best_metric:.6f}）")
                                writer.close()
                                if is_dist: torch.distributed.destroy_process_group()
                                return
            model_eval.train()

        # 按 epoch 保存一份
        if is_main(rank):
            save_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            save_ckpt(save_model, optimizer, epoch, step, log_dir, f'epoch_{epoch+1}.pth')

    if is_main(rank) and writer is not None:
        writer.close()
    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
