import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time

from dnn.torch.fargan.dataset import FARGANDataset
from dnn.torch.fargan.stft_loss import MultiResolutionSTFTLoss
from dnn.FarGanRVQ.models.optimized_fargan import OptimizedFarGan

# import OSCE discriminators
import sys
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../..", "torch", "osce"))
import models as osce_models


def fmap_loss(scores_real, scores_gen):
	loss_feat = 0
	num_discs = len(scores_real)
	for k in range(num_discs):
		num_layers = len(scores_gen[k]) - 1
		f = 4 / num_discs / num_layers
		for l in range(num_layers):
			loss_feat += f * F.l1_loss(scores_gen[k][l], scores_real[k][l].detach())
	return loss_feat


def main():
	parser = argparse.ArgumentParser(description='FarGanRVQ Adversarial Training (minimal)')
	parser.add_argument('features', type=str)
	parser.add_argument('signal', type=str)
	parser.add_argument('output', type=str)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--sequence-length', type=int, default=60)
	parser.add_argument('--lr', type=float, default=5e-4)
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

	dataset = FARGANDataset(args.features, args.signal, sequence_length=args.sequence_length)
	dl = torch.utils.data.DataLoader(
		dataset, batch_size=args.batch_size, shuffle=True, 
		drop_last=args.drop_last, num_workers=num_workers, pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		prefetch_factor=args.prefetch_factor if num_workers > 0 else None
	)

	model = OptimizedFarGan().to(device)
	disc = osce_models.model_dict['fdmresdisc'](
		architecture='free', design='f_down',
		fft_sizes_16k=[2**n for n in range(6, 12)], freq_roi=[0, 7400],
		max_channels=256, noise_gain=0.0
	).to(device)

	opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
	opt_d = torch.optim.AdamW([p for p in disc.parameters() if p.requires_grad], lr=args.lr, betas=(0.8, 0.99))
	spec_loss = MultiResolutionSTFTLoss(device).to(device)
	
	# 混合精度训练
	scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

	last_end_time = time.time()
	for epoch in range(1, args.epochs + 1):
		with tqdm.tqdm(dl, unit='batch') as tepoch:
			for i, (features, periods, target, lpc) in enumerate(tepoch):
				# 统计数据加载耗时
				now_time = time.time()
				data_time = now_time - last_end_time
				
				features = features.to(device, non_blocking=True)
				target = target[:, :args.sequence_length*160].to(device, non_blocking=True)
				features = features[:, :args.sequence_length+4, :]

				# simple warmup: pre=2 frames
				nb_pre = 2
				pre = target[:, :nb_pre*160]
				
				if args.mixed_precision and device.type == 'cuda':
					with torch.amp.autocast('cuda'):
						pred = model(features)
						output = torch.cat([pre, pred[:, nb_pre*160:args.sequence_length*160]], dim=-1)

						# update D
						scores_gen = disc(output.detach().unsqueeze(1))
						scores_real = disc(target.unsqueeze(1))
						d_loss = 0
						for scale in scores_gen:
							d_loss += ((scale[-1]) ** 2).mean()
						for scale in scores_real:
							d_loss += ((1 - scale[-1]) ** 2).mean()
						d_loss = 0.5 * d_loss / len(scores_gen)
					
					opt_d.zero_grad(set_to_none=True)
					scaler.scale(d_loss).backward()
					scaler.step(opt_d)
					
					with torch.amp.autocast('cuda'):
						# update G
						scores_gen = disc(output.unsqueeze(1))
						cont = spec_loss(output, target.detach())
						g_loss_adv = 0
						for scale in scores_gen:
							g_loss_adv += ((1 - scale[-1]) ** 2).mean() / len(scores_gen)
						feat = fmap_loss(scores_real, scores_gen)
						g_loss = cont + feat + g_loss_adv
					
					scaler.scale(g_loss).backward()
					scaler.step(opt_g)
					scaler.update()
					opt_g.zero_grad(set_to_none=True)
				else:
					pred = model(features)
					output = torch.cat([pre, pred[:, nb_pre*160:args.sequence_length*160]], dim=-1)

					# update D
					scores_gen = disc(output.detach().unsqueeze(1))
					scores_real = disc(target.unsqueeze(1))
					d_loss = 0
					for scale in scores_gen:
						d_loss += ((scale[-1]) ** 2).mean()
					for scale in scores_real:
						d_loss += ((1 - scale[-1]) ** 2).mean()
					d_loss = 0.5 * d_loss / len(scores_gen)
					opt_d.zero_grad(set_to_none=True)
					d_loss.backward()
					opt_d.step()

					# update G
					scores_gen = disc(output.unsqueeze(1))
					cont = spec_loss(output, target.detach())
					g_loss_adv = 0
					for scale in scores_gen:
						g_loss_adv += ((1 - scale[-1]) ** 2).mean() / len(scores_gen)
					feat = fmap_loss(scores_real, scores_gen)
					g_loss = cont + feat + g_loss_adv
					opt_g.zero_grad(set_to_none=True)
					g_loss.backward()
					opt_g.step()
				
				# 统计计算耗时
				end_time = time.time()
				compute_time = end_time - now_time
				last_end_time = end_time

				postfix = {"d_loss": f"{d_loss.item():.4f}", "g_loss": f"{g_loss.item():.4f}"}
				if args.log_timing:
					postfix['data'] = f'{data_time*1000:.1f}ms'
					postfix['compute'] = f'{compute_time*1000:.1f}ms'
				tepoch.set_postfix(**postfix)

		ckpt = {
			'state_dict': model.state_dict(),
			'epoch': epoch
		}
		torch.save(ckpt, os.path.join(args.output, 'checkpoints', f'far_gan_rvq_adv_{epoch}.pth'))


if __name__ == '__main__':
	main() 