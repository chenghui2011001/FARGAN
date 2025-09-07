import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import time

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.enhanced_fargan import EnhancedFarGan
from dnn.FarGanRVQ.models.enhanced_losses import EnhancedCompositeLoss

# Import OSCE discriminators
import sys
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../..", "torch", "osce"))
import models as osce_models


def main():
	parser = argparse.ArgumentParser(description='Enhanced FarGan Training with Optimizations')
	parser.add_argument('features', type=str)
	parser.add_argument('signal', type=str)
	parser.add_argument('output', type=str)
	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--sequence-length', type=int, default=60)
	parser.add_argument('--lr-g', type=float, default=3e-5)
	parser.add_argument('--lr-d', type=float, default=1e-4)
	parser.add_argument('--warmup-steps', type=int, default=2000)
	# 新增：性能/并行相关参数
	parser.add_argument('--num-workers', type=int, default=-1, help='DataLoader 工作线程数，-1 表示自动')
	parser.add_argument('--prefetch-factor', type=int, default=2, help='每个 worker 预取批次数（num_workers>0 时生效）')
	parser.add_argument('--drop-last', action='store_true', default=True, help='丢弃最后一个不足批量的批次')
	parser.add_argument('--grad-accum-steps', type=int, default=1, help='梯度累积步数，用于放大有效批量')
	parser.add_argument('--enable-tf32', action='store_true', help='在支持的 GPU 上启用 TF32 加速')
	parser.add_argument('--log-timing', action='store_true', help='记录数据加载/计算耗时拆分')
	parser.add_argument('--mixed-precision', action='store_true', help='启用混合精度训练')
	args = parser.parse_args()

	os.makedirs(os.path.join(args.output, 'checkpoints'), exist_ok=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"🚀 使用设备: {device}")
	
	# 可选：启用 TF32 加速（Ampere+）
	if device.type == 'cuda' and args.enable_tf32:
		try:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
			print("⚙️ 已启用 TF32 加速 (matmul/cudnn)")
		except Exception as e:
			print(f"⚠️ 启用 TF32 失败: {e}")
	
	# 数据加载器优化
	if args.num_workers < 0:
		num_workers = 4 if device.type == 'cuda' else 2
	else:
		num_workers = args.num_workers
	pin_memory = device.type == 'cuda'
	persistent_workers = num_workers > 0

	# Dataset
	dataset = FARGANDataset(args.features, args.signal, sequence_length=args.sequence_length)
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=args.batch_size, shuffle=True, 
		drop_last=args.drop_last, num_workers=num_workers, pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		prefetch_factor=args.prefetch_factor if num_workers > 0 else None
	)

	# Enhanced model
	model = EnhancedFarGan(in_features=20, cond_dim=32, subframe_size=40).to(device)
	
	# Multiple discriminators for better coverage
	disc_stft = osce_models.model_dict['fdmresdisc'](
		architecture='free', design='f_down',
		fft_sizes_16k=[2**n for n in range(6, 12)], 
		freq_roi=[0, 7400], max_channels=256, noise_gain=0.0
	).to(device)
	
	disc_wave = osce_models.model_dict['fdmresdisc'](
		architecture='free', design='f_down',
		fft_sizes_16k=[2**n for n in range(7, 11)],  # Different scales
		freq_roi=[0, 8000], max_channels=128, noise_gain=0.1
	).to(device)

	# Enhanced loss
	composite_loss = EnhancedCompositeLoss(sample_rate=16000).to(device)

	# Optimizers with different learning rates
	opt_g = torch.optim.AdamW(
		list(model.parameters()) + list(composite_loss.parameters()),
		lr=args.lr_g, betas=(0.8, 0.99), eps=1e-8, weight_decay=1e-4
	)
	
	opt_d = torch.optim.AdamW(
		list(disc_stft.parameters()) + list(disc_wave.parameters()),
		lr=args.lr_d, betas=(0.8, 0.99), eps=1e-8
	)

	# Learning rate schedulers
	scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		opt_g, T_0=5000, T_mult=2, eta_min=1e-6
	)
	scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		opt_d, T_0=5000, T_mult=2, eta_min=1e-6
	)

	# Mixed precision training
	scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

	step = 0
	last_end_time = time.time()
	for epoch in range(1, args.epochs + 1):
		print(f"\nEpoch {epoch}/{args.epochs}")
		
		with tqdm.tqdm(dataloader, unit='batch') as tepoch:
			for i, (features, periods, target, lpc) in enumerate(tepoch):
				# 统计数据加载耗时
				now_time = time.time()
				data_time = now_time - last_end_time
				
				step += 1
				
				# Move to device
				features = features.to(device, non_blocking=True)
				periods = periods.to(device, non_blocking=True) 
				target = target[:, :args.sequence_length*160].to(device, non_blocking=True)
				features = features[:, :args.sequence_length+4, :]
				periods = periods[:, :args.sequence_length+4]

				# Extract voicing from features (if available, else use dummy)
				voicing = torch.ones_like(periods[:, :args.sequence_length])  # Placeholder

				with torch.amp.autocast('cuda', enabled=scaler is not None):
					# Generate output
					output = model(features, periods[:, :args.sequence_length], voicing)
					
					# Ensure output length matches target
					min_len = min(output.size(1), target.size(1))
					output = output[:, :min_len]
					target_trim = target[:, :min_len]

					# ===== Discriminator Update =====
					# STFT discriminator
					scores_real_stft = disc_stft(target_trim.unsqueeze(1))
					scores_fake_stft = disc_stft(output.detach().unsqueeze(1))
					
					# Wave discriminator  
					scores_real_wave = disc_wave(target_trim.unsqueeze(1))
					scores_fake_wave = disc_wave(output.detach().unsqueeze(1))

					# Discriminator loss
					d_loss = 0
					for scale_real, scale_fake in zip(scores_real_stft, scores_fake_stft):
						d_loss += ((1 - scale_real[-1]) ** 2).mean()
						d_loss += (scale_fake[-1] ** 2).mean()
					
					for scale_real, scale_fake in zip(scores_real_wave, scores_fake_wave):
						d_loss += ((1 - scale_real[-1]) ** 2).mean()
						d_loss += (scale_fake[-1] ** 2).mean()
					
					d_loss = d_loss / (len(scores_real_stft) + len(scores_real_wave))

				# Discriminator backward
				opt_d.zero_grad()
				if scaler:
					scaler.scale(d_loss).backward()
					scaler.step(opt_d)
				else:
					d_loss.backward()
					opt_d.step()

				with torch.amp.autocast('cuda', enabled=scaler is not None):
					# ===== Generator Update =====
					scores_fake_stft = disc_stft(output.unsqueeze(1))
					scores_fake_wave = disc_wave(output.unsqueeze(1))
					
					# Enhanced composite loss
					g_loss, loss_dict = composite_loss(
						pred=output,
						target=target_trim,
						periods=periods[:, :args.sequence_length],
						real_features=scores_real_stft + scores_real_wave,
						fake_features=scores_fake_stft + scores_fake_wave,
						fake_scores=scores_fake_stft + scores_fake_wave,
						mode='generator'
					)

				# Generator backward with gradient clipping
				opt_g.zero_grad()
				if scaler:
					scaler.scale(g_loss).backward()
					scaler.unscale_(opt_g)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					scaler.step(opt_g)
					scaler.update()
				else:
					g_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					opt_g.step()

				# Update learning rates
				scheduler_g.step()
				scheduler_d.step()
				
				# 统计计算耗时
				end_time = time.time()
				compute_time = end_time - now_time
				last_end_time = end_time

				# Logging
				log_dict = {
					'd_loss': f"{d_loss.item():.4f}",
					'g_loss': f"{g_loss.item():.4f}",
					'lr_g': f"{opt_g.param_groups[0]['lr']:.1e}",
				}
				
				# Add individual loss components
				for k, v in loss_dict.items():
					if k != 'total':
						log_dict[k] = f"{v.item():.4f}"
				
				# 添加计时信息
				if args.log_timing:
					log_dict['data'] = f'{data_time*1000:.1f}ms'
					log_dict['compute'] = f'{compute_time*1000:.1f}ms'

				tepoch.set_postfix(**log_dict)

				# Save intermediate checkpoints
				if step % 2000 == 0:
					torch.save({
						'model': model.state_dict(),
						'opt_g': opt_g.state_dict(),
						'opt_d': opt_d.state_dict(),
						'step': step,
						'loss_weights': {k: v.item() for k, v in composite_loss.named_parameters()}
					}, os.path.join(args.output, 'checkpoints', f'enhanced_fargan_step_{step}.pth'))

		# End of epoch save
		torch.save({
			'model': model.state_dict(),
			'disc_stft': disc_stft.state_dict(),
			'disc_wave': disc_wave.state_dict(),
			'composite_loss': composite_loss.state_dict(),
			'opt_g': opt_g.state_dict(),
			'opt_d': opt_d.state_dict(),
			'epoch': epoch,
			'step': step
		}, os.path.join(args.output, 'checkpoints', f'enhanced_fargan_epoch_{epoch}.pth'))

		print(f"Epoch {epoch} completed. Model saved.")


if __name__ == '__main__':
	main() 