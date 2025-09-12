#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单卡适配版预训练入口（不修改原脚本）：
- 通过轻量 monkey patch 让 torch.cuda.set_device 接受 'cuda'（无索引）并映射到 0；
- 显式清除 DDP 相关环境变量，强制单进程/单卡；
- 直接调用原脚本 train_pretrain_mamba.main()，所有参数保持一致（含 --resume/--ar-preheat）。

用法示例：
  CUDA_VISIBLE_DEVICES=0 python dnn/FarGanRVQ/training/train_pretrain_mamba_single.py \
    --features data_cn/out_features.f32 --pcm data_cn/out_speech.pcm \
    --epochs 3 --batch-size 32 --parallel-train --mixed-precision
"""

import os
import sys
import torch


def _patch_cuda_set_device():
    import torch.cuda as _cuda
    _orig = _cuda.set_device

    def _set_device_fix(dev):
        try:
            # 允许 dev 为 'cuda' 或 torch.device('cuda')（无索引），映射到 0
            if isinstance(dev, str) and dev == 'cuda':
                return _orig(0)
            if isinstance(dev, torch.device) and dev.type == 'cuda' and dev.index is None:
                return _orig(0)
            return _orig(dev)
        except Exception:
            # 兜底：退回设备 0
            return _orig(0)

    _cuda.set_device = _set_device_fix  # type: ignore


def _clear_ddp_env():
    for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        if k in os.environ:
            os.environ.pop(k, None)


def main():
    # 单卡环境准备
    _clear_ddp_env()
    _patch_cuda_set_device()

    # 保持与原脚本相同的 sys.path 行为
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    fargan_path = os.path.join(project_root, 'dnn/torch/fargan')
    if fargan_path not in sys.path:
        sys.path.insert(0, fargan_path)

    # 调用原脚本入口
    from dnn.FarGanRVQ.training import train_pretrain_mamba as _orig
    _orig.main()


if __name__ == '__main__':
    main()

