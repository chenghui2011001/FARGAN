import argparse
import torch
from torch.utils.data import DataLoader
from dnn.torch.fargan.dataset import FARGANDataset
from dnn.FarGanRVQ.models.baseline import FeatureToWaveMLP


def collate_fn(batch):
	features, periods, data, lpc = zip(*batch)
	features = torch.tensor(features, dtype=torch.float32)
	data = torch.tensor(data, dtype=torch.float32)
	return features, data


def main():
	parser = argparse.ArgumentParser(description='FarGanRVQ Minimal Evaluation')
	parser.add_argument('--features', type=str, required=True)
	parser.add_argument('--pcm', type=str, required=True)
	parser.add_argument('--ckpt', type=str, required=True)
	args = parser.parse_args()

	ds = FARGANDataset(args.features, args.pcm)
	dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

	T = 15 * 2 + 4
	F_used = 20
	N = 160 * 15
	model = FeatureToWaveMLP(feature_dim=F_used, time_steps=T, out_samples=N)
	state = torch.load(args.ckpt, map_location='cpu')
	model.load_state_dict(state['model'])
	model.eval()

	import torch.nn as nn
	loss_fn = nn.L1Loss(reduction='sum')
	total, count = 0.0, 0
	with torch.no_grad():
		for i, (features, data) in enumerate(dl):
			y = data[:, :N]
			y_hat = model(features)
			total += loss_fn(y_hat, y).item()
			count += y.numel()
			if i >= 10:
				break

	mae = total / count
	print(f'MAE over {count} samples: {mae:.6f}')


if __name__ == '__main__':
	main() 