import torch
import torch.nn as nn

class FeatureToWaveMLP(nn.Module):
	"""Very small MLP: flatten features (B,T,F) -> waveform chunk (B,N)."""
	def __init__(self, feature_dim: int, time_steps: int, out_samples: int, hidden: int = 256):
		super().__init__()
		self.in_dim = feature_dim * time_steps
		self.out_dim = out_samples
		self.net = nn.Sequential(
			nn.Linear(self.in_dim, hidden),
			nn.GELU(),
			nn.Linear(hidden, self.out_dim)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, t, f = x.shape
		x = x.reshape(b, t * f)
		return self.net(x) 