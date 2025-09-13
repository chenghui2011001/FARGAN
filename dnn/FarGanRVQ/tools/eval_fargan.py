#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立评测脚本（不改训练）
- 加载 MambaEnhancedFarGan 最新 ckpt
- 用 FARGANDataset 跑少量样本
- 修正采样率/反归一化/去预加重/维度
- 以新 tag 写入 TensorBoard：默认 Val_fixed/pred, Val_fixed/target
"""

import os, sys, argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import soundfile as sf


# ---------- 路径与导入（与训练脚本保持一致） ----------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
fargan_path = os.path.join(project_root, 'dnn/torch/fargan')
sys.path.insert(0, fargan_path)

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.mamba_enhanced_fargan import MambaEnhancedFarGan

# ---------- 与训练一致的 collate ----------
def collate_fn(batch):
    feats, pers, data = [], [], []
    for features, periods, waveform, lpc in batch:
        feats.append(features)
        pers.append(periods)
        data.append(waveform)
    feats = torch.from_numpy(np.array(feats)).float()
    pers  = torch.from_numpy(np.array(pers)).float()
    data  = torch.from_numpy(np.array(data)).float()
    return feats, pers, data

# ---------- 数值修正工具 ----------
def denorm_none(x): return x
def denorm_meanstd(x, mean=0.0, std=1.0): return x * std + mean
def denorm_minmax(x, minv=-1.0, maxv=1.0): return x  # 如有自定义缩放，请在此还原

def deemph(x, p=0.97):
    # 去预加重；x: [B,T] 或 [T]
    if x.ndim == 1:
        y = torch.zeros_like(x)
        y[0] = x[0]
        for t in range(1, x.shape[0]):
            y[t] = x[t] + p * y[t-1]
        return y
    elif x.ndim == 2:
        B, T = x.shape
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0]
        for t in range(1, T):
            y[:, t] = x[:, t] + p * y[:, t-1]
        return y
    else:
        raise ValueError("deemph expects 1D or 2D tensor")

def build_model(F_used=20, period_shift=3, output_delay=0):
    # 与训练脚本构造保持一致
    return MambaEnhancedFarGan(in_features=F_used, cond_dim=32, subframe_size=40,
                               cond_shift=2, period_shift=period_shift,
                               output_delay=output_delay)

def build_dataset(args, frame_size=160, seq_len=60, F_used=20):
    return FARGANDataset(
        feature_file=args.features,
        signal_file=args.pcm,
        frame_size=frame_size,
        sequence_length=seq_len,
        lookahead=1,
        nb_used_features=F_used,
        nb_features=36,
    )

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    # 数据路径（与训练相同的二进制输入）
    ap.add_argument("--features", type=str, required=True, help="特征二进制路径 (.f32)")
    ap.add_argument("--pcm",      type=str, required=True, help="目标波形二进制路径 (.pcm int16 16kHz)")

    # 模型与评测
    ap.add_argument("--ckpt", type=str, required=True, help="生成器 ckpt(.pth) 路径")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=2, help="写入多少条到 TensorBoard")
    ap.add_argument("--seq-len", type=int, default=60, help="与训练一致的序列长度")
    ap.add_argument("--F-used",  type=int, default=20, help="与训练一致的 nb_used_features")
    ap.add_argument("--outdir", type=str, default="runs_eval")
    ap.add_argument("--period-shift", type=int, default=3, help="模型周期偏移 period_shift（默认3）")
    ap.add_argument("--output-delay", type=int, default=0, help=">0 延后，<0 提前，对生成结果做样本级补偿")
    ap.add_argument("--tag", type=str, default="Val_fixed", help="根 tag（如 Val_fixed 或 Test）")

    # 播放/数值修正
    ap.add_argument("--sr", type=int, default=16000, help="写入到 TensorBoard 的采样率")
    ap.add_argument("--denorm-mode", choices=["none", "meanstd", "minmax"], default="none")
    ap.add_argument("--denorm-mean", type=float, default=0.0)
    ap.add_argument("--denorm-std",  type=float, default=1.0)
    ap.add_argument("--denorm-min",  type=float, default=-1.0)
    ap.add_argument("--denorm-max",  type=float, default=1.0)
    ap.add_argument("--deemph", type=float, default=0.0, help=">0 启用去预加重，如 0.97")
    ap.add_argument("--teacher-forcing", action="store_true",
                help="评测时以 target 作为 teacher_signal 旁路（仅诊断）")
    ap.add_argument("--ar-preheat", type=int, default=0,
                help="自回归评测时的预热帧数（每帧160样本）。>0 时将使用 target 的前 N 帧做激励记忆预热，仅在未启用 --teacher-forcing 时生效")
    ap.add_argument("--rms-match", action="store_true",
                help="写入前把预测 RMS 对齐到 target（只用于听感验证）")
    ap.add_argument("--print-metrics", action="store_true",
                    help="打印对齐后的 lag/SNR/SI-SDR 诊断信息")


    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        # 兼容旧版 torch 没有 weights_only 参数
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model = build_model(F_used=args.F_used, period_shift=args.period_shift,
                        output_delay=args.output_delay).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[CKPT] missing:", len(missing), "unexpected:", len(unexpected))
    if len(missing) > 0:
        print(missing[:20])
    if missing:   print("[WARN] missing keys:", len(missing))
    if unexpected:print("[WARN] unexpected keys:", len(unexpected))
    model.eval()

    # 数据
    ds = build_dataset(args, frame_size=160, seq_len=args.seq_len, F_used=args.F_used)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
        persistent_workers=(args.num_workers>0), collate_fn=collate_fn
    )

    # 选择反归一化函数（注意这里全部是下划线属性名）
    if args.denorm_mode == "none":
        fn_denorm = denorm_none
    elif args.denorm_mode == "meanstd":
        fn_denorm = lambda x: denorm_meanstd(x, args.denorm_mean, args.denorm_std)
    else:
        fn_denorm = lambda x: denorm_minmax(x, args.denorm_min, args.denorm_max)

    # TensorBoard
    run_name = f"{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(args.outdir, run_name))
    save_dir = os.path.join(args.outdir, run_name, "wavs")
    os.makedirs(save_dir, exist_ok=True)
    written, global_step = 0, 0
    for feats, periods, target in dl:
        feats   = feats.to(device, non_blocking=True)
        periods = periods.to(device, non_blocking=True)
        target  = target.to(device, non_blocking=True)       # [B, T]
        # 生成预测；对齐目标长度
        # Teacher Forcing：并行子帧路径
        if args.teacher_forcing:
            y_hat = model(
                feats,
                periods=periods,
                csi=None, channel_noise=None,
                target_length=min(target.shape[1], feats.shape[1]*160),
                parallel_train=True,
                teacher_signal=target,
            )
        else:
            # AR 路径：可选预热（以 target 的前 N 帧初始化激励记忆）
            preheat = None
            if args.ar_preheat and args.ar_preheat > 0:
                N = int(args.ar_preheat) * 160
                preheat = target[:, :min(N, target.shape[1])]
            y_hat = model(
                feats,
                periods=periods,
                csi=None, channel_noise=None,
                target_length=min(target.shape[1], feats.shape[1]*160),
                parallel_train=False,
                teacher_signal=preheat,
            )

        y_hat = y_hat[:, :target.shape[1]]                    # [B, T]

        # 移到 CPU 做数值修正
        y_hat = y_hat.detach().cpu().float()
        target = target.detach().cpu().float()

        # 反归一化
        y_hat = fn_denorm(y_hat)
        target = fn_denorm(target)

        # 去预加重（如需要）
        if args.deemph and args.deemph > 0:
            y_hat = deemph(y_hat, p=args.deemph)
            target = deemph(target, p=args.deemph)

        # 限幅到 [-1,1]
        y_hat.clamp_(-1.0, 1.0)
        # --- 可选：匹配每条样本的 RMS（只用于听感验证，不回写模型） ---
        # def rms(x): return (x.pow(2).mean().sqrt() + 1e-8)
        # scale = (rms(target) / rms(y_hat)).item()
        # scale = max(0.5, min(scale, 2.0))  # 防疯
        # y_hat = (y_hat * scale).clamp_(-1.0, 1.0)

        target.clamp_(-1.0, 1.0)
        # 可选：RMS 匹配（只为听感核验）
        if args.rms_match:
            def _rms(x): return (x.pow(2).mean().sqrt() + 1e-8)
            scale = float(_rms(target) / _rms(y_hat))
            scale = max(0.5, min(scale, 2.0))
            y_hat = (y_hat * scale).clamp_(-1.0, 1.0)

        # 可选：打印诊断指标
        if args.print_metrics:
            import torch.nn.functional as F
            def _best_lag_1d(x, y, max_lag=800):
                best_lag, best_c = 0, -1e9
                for L in range(-max_lag, max_lag+1):
                    if L >= 0:
                        c = F.cosine_similarity(x[L:], y[:y.numel()-L], dim=0)
                    else:
                        L2 = -L
                        c = F.cosine_similarity(x[:x.numel()-L2], y[L2:], dim=0)
                    val = float(c)
                    if val > best_c:
                        best_c, best_lag = val, L
                return best_lag
            def _align(x, y, lag):
                if lag > 0: return x[lag:], y[:y.numel()-lag]
                if lag < 0: return x[:x.numel()+lag], y[-lag:]
                return x, y
            def _snr_db(a, b, eps=1e-8):
                return 10*torch.log10((b.pow(2).mean()+eps)/((b-a).pow(2).mean()+eps))
            def _si_sdr(a, b, eps=1e-8):
                s_target = (torch.sum(a*b) / (torch.sum(b*b)+eps)) * b
                e_noise  = a - s_target
                return 10*torch.log10((torch.sum(s_target**2)+eps)/(torch.sum(e_noise**2)+eps))
            y1 = y_hat[0].clone(); t1 = target[0].clone()
            lag = _best_lag_1d(y1, t1, max_lag=800)
            ya, ta = _align(y1, t1, lag)
            L = min(ya.numel(), ta.numel()); ya=ya[:L]; ta=ta[:L]
            print(f"[diag] lag={lag} samples (~{lag/16000:.4f}s)  SNR={_snr_db(ya,ta):.2f} dB  SI-SDR={_si_sdr(ya,ta):.2f} dB")

        # 打印一条范围，便于快速确认
        print(f"[step {global_step}] eval_mode=True  pred_range=({y_hat.min():.3f},{y_hat.max():.3f})  "
              f"target_range=({target.min():.3f},{target.max():.3f})  sr={args.sr}")
        # 放在写入循环前、y_hat/target 都是 [B,T] 的时刻
        import torch.nn.functional as F

        def snr_db(x, y):
            num = (y**2).mean()
            den = ((y - x)**2).mean() + 1e-12
            return 10 * torch.log10(num / den)

        def best_lag(x, y, max_lag=800):  # 50ms @16k
            # 取单条，粗暴滑窗互相关
            x1, y1 = x[0], y[0]
            lags = torch.arange(-max_lag, max_lag+1)
            best, best_c = 0, -1e9
            for L in lags:
                if L >= 0:
                    c = F.cosine_similarity(x1[L:], y1[:y1.numel()-L], dim=0)
                else:
                    L2 = -L
                    c = F.cosine_similarity(x1[:x1.numel()-L2], y1[L2:], dim=0)
                if c.item() > best_c:
                    best_c, best = c.item(), int(L.item())
            return best, best_c

        # 你已有的：估计整批 lag（用第 1 条）
        lag, corr = best_lag(y_hat, target, max_lag=800)

        # === 新增：按整批 lag 对齐（只做诊断/听感用，不回写模型） ===
        if lag > 0:
            y_hat  = y_hat[:, lag:]
            target = target[:, :y_hat.shape[1]]
        elif lag < 0:
            L2 = -lag
            target = target[:, L2:]
            y_hat  = y_hat[:, :target.shape[1]]

        # === 改：对齐后再算一次并打印 ===
        print(f"[diag-aligned] lag={lag} (~{lag/16000:.3f}s), "
            f"cos={corr:.3f}, snr={snr_db(y_hat, target):.2f} dB")

        # 逐条写入
        B = y_hat.shape[0]
        for i in range(B):
            if written >= args.num_samples:
                break
            p1d = y_hat[i].contiguous().view(-1)     # [T]
            t1d = target[i].contiguous().view(-1)    # [T]
            writer.add_audio(f"{args.tag}/pred",   p1d, global_step=global_step, sample_rate=args.sr)
            writer.add_audio(f"{args.tag}/target", t1d, global_step=global_step, sample_rate=args.sr)
            sf.write(os.path.join(save_dir, f"pred_{global_step:03d}.wav"),
                    p1d.numpy(), args.sr)
            sf.write(os.path.join(save_dir, f"tgt_{global_step:03d}.wav"),
                    t1d.numpy(), args.sr)
            written += 1
            global_step += 1
        if written >= args.num_samples:
            break

    writer.flush(); writer.close()
    print(f"Done. Wrote {written} samples under '{args.tag}/*'. Logdir: {os.path.join(args.outdir, run_name)}")

if __name__ == "__main__":
    main()
