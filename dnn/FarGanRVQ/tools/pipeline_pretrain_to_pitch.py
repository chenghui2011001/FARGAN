#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import glob
import shutil
import subprocess
from datetime import datetime

PYTHON = sys.executable  # 当前环境的 python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

def run(cmd, env=None):
    print("\n==== RUN ====\n" + " ".join(cmd) + "\n==============\n", flush=True)
    p = subprocess.Popen(cmd, env=env)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")

def latest_subdir(parent, prefix):
    """
    在 parent 下找以 prefix 开头的子目录，按 mtime 逆序返回最新的一个路径；找不到返回 None
    """
    if not os.path.isdir(parent):
        return None
    cands = [os.path.join(parent, d) for d in os.listdir(parent)
             if os.path.isdir(os.path.join(parent, d)) and d.startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
    return cands[0]

def find_best_or_last_ckpt(run_dir, best_name='best.pth', ep_glob='pretrain_epoch_*.pth'):
    """
    优先找 best.pth；否则取 ep_glob 中按数字最大的一个；找不到返回 None
    """
    best_path = os.path.join(run_dir, best_name)
    if os.path.isfile(best_path):
        return best_path
    eps = sorted(glob.glob(os.path.join(run_dir, ep_glob)))
    if not eps:
        return None
    # 取最大的 epoch 号
    def _epn(p):
        bn = os.path.basename(p)
        # pretrain_epoch_12.pth
        try:
            return int(bn.split('_')[-1].split('.')[0])
        except Exception:
            return -1
    eps.sort(key=_epn, reverse=True)
    return eps[0]

def main():
    ap = argparse.ArgumentParser(description="A(预训练) ➜ B(pitch 微调) 自动流水线")
    # 数据与设备
    ap.add_argument('--features', required=True, help='data/out_features.f32')
    ap.add_argument('--pcm',      required=True, help='data/out_speech.pcm')
    ap.add_argument('--devices',  default='0,1', help='CUDA_VISIBLE_DEVICES，例如 "0,1" 或 "0"')
    ap.add_argument('--nproc',    type=int, default=2, help='每台卡数')
    
    # A 阶段（预训练）配置
    ap.add_argument('--a_outdir', default='dnn/FarGanRVQ/checkpoints/pretrain_visible')
    ap.add_argument('--a_epochs', type=int, default=50)
    ap.add_argument('--a_batch',  type=int, default=1024)
    ap.add_argument('--a_seq',    type=int, default=60)
    ap.add_argument('--a_stft',   type=int, nargs='+', default=[1024])
    ap.add_argument('--a_disable_phase', action='store_true', default=True)
    ap.add_argument('--a_parallel_train', action='store_true', default=True)
    ap.add_argument('--a_num_workers', type=int, default=16)
    ap.add_argument('--a_prefetch',    type=int, default=6)
    ap.add_argument('--a_drop_last',   action='store_true', default=True)
    ap.add_argument('--a_logint',      type=int, default=100)
    ap.add_argument('--a_mixed',       action='store_true', default=True)

    # B 阶段（polish：只解冻 pitch）配置
    ap.add_argument('--b_outdir', default='dnn/FarGanRVQ/checkpoints/pretrain_stageB_pitch')
    ap.add_argument('--b_epochs', type=int, default=5)
    ap.add_argument('--b_batch',  type=int, default=1024)
    ap.add_argument('--b_seq',    type=int, default=60)
    ap.add_argument('--b_lr',     type=float, default=1e-5)
    ap.add_argument('--b_stft',   type=int, nargs='+', default=[1024])
    ap.add_argument('--b_disable_phase', action='store_true', default=True)
    ap.add_argument('--b_parallel_train', action='store_true', default=True)
    ap.add_argument('--b_num_workers', type=int, default=16)
    ap.add_argument('--b_prefetch',    type=int, default=6)
    ap.add_argument('--b_drop_last',   action='store_true', default=True)
    ap.add_argument('--b_logint',      type=int, default=100)
    ap.add_argument('--b_mixed',       action='store_true', default=True)

    ap.add_argument('--master_port', type=str, default='29512', help='torchrun master_port')
    ap.add_argument('--dry_run', action='store_true', help='仅打印命令，不真正执行')
    args = ap.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.devices

    # ========== A 阶段：预训练 ==========
    a_cmd = [
        'torchrun', '--nproc_per_node', str(args.nproc), '--master_port', args.master_port,
        os.path.join(PROJECT_ROOT, 'dnn/FarGanRVQ/training/train_pretrain_mamba.py'),
        '--features', args.features,
        '--pcm',      args.pcm,
        '--batch-size', str(args.a_batch),
        '--seq-len',    str(args.a_seq),
        '--epochs',     str(args.a_epochs),
        '--num-workers', str(args.a_num_workers),
        '--prefetch-factor', str(args.a_prefetch),
        '--log-interval', str(args.a_logint),
        '--outdir', args.a_outdir,
        '--stft-sizes', *[str(x) for x in args.a_stft],
    ]
    if args.a_disable_phase:
        a_cmd.append('--disable-phase-loss')
    if args.a_parallel_train:
        a_cmd.append('--parallel-train')
    if args.a_drop_last:
        a_cmd.append('--drop-last')
    if args.a_mixed:
        a_cmd.append('--mixed-precision')

    print("\n### [Stage A] 预训练 ###")
    print("Log dir base:", args.a_outdir)
    if not args.dry_run:
        run(a_cmd, env=env)

    # 找最新 pretrain 目录（形如 pretrain_mamba_YYYYMMDD_HHMMSS）
    latest_run = latest_subdir(args.a_outdir, prefix='pretrain_mamba_')
    if latest_run is None:
        raise RuntimeError(f"未在 {args.a_outdir} 下找到 pretrain_mamba_* 目录")
    print("[A] latest pretrain run:", latest_run)

    best_ckpt = find_best_or_last_ckpt(latest_run, best_name='best.pth', ep_glob='pretrain_epoch_*.pth')
    if best_ckpt is None:
        raise RuntimeError(f"在 {latest_run} 下找不到 best.pth 或 pretrain_epoch_*.pth")
    print("[A] 作为 B 阶段的启动 ckpt:", best_ckpt)

    # ========== B 阶段：只解冻 pitch 的 polish ==========
    b_cmd = [
        'torchrun', '--nproc_per_node', str(args.nproc), '--master_port', args.master_port,
        os.path.join(PROJECT_ROOT, 'dnn/FarGanRVQ/training/train_polish_mamba.py'),
        '--features', args.features,
        '--pcm',      args.pcm,
        '--resume',   best_ckpt,
        '--seq-len',  str(args.b_seq),
        '--batch-size', str(args.b_batch),
        '--epochs',   str(args.b_epochs),
        '--lr',       str(args.b_lr),
        '--num-workers', str(args.b_num_workers),
        '--prefetch-factor', str(args.b_prefetch),
        '--log-interval', str(args.b_logint),
        '--outdir', args.b_outdir,
        '--stft-sizes', *[str(x) for x in args.b_stft],
        '--unfreeze-pitch',   # 关键：只解冻 pitch_predictor
    ]
    if args.b_disable_phase:
        b_cmd.append('--disable-phase-loss')
    if args.b_parallel_train:
        b_cmd.append('--parallel-train')
    if args.b_drop_last:
        b_cmd.append('--drop-last')
    if args.b_mixed:
        b_cmd.append('--mixed-precision')

    print("\n### [Stage B] polish(仅 pitch) ###")
    print("Log dir base:", args.b_outdir)
    if not args.dry_run:
        run(b_cmd, env=env)

    print("\n✅ Pipeline 完成：A(预训练) ➜ B(polish) 已结束。\n")

if __name__ == '__main__':
    main()
