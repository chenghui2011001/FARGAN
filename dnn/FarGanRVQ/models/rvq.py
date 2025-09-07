import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRVQ(nn.Module):
	"""Minimal residual vector quantizer with straight-through estimator.
	- L stages with codebook sizes.
	- Input: [B,T,D_in] -> project to D (shared), quantize per stage with residual.
	- Returns: cond [B,T,D], indices [L,B,T], vq_loss (commitment + codebook)."""
	def __init__(self, d_in: int = 64, d_code: int = 32, codebook_sizes=(128, 64), beta: float = 0.25):
		super().__init__()
		self.proj_in = nn.Linear(d_in, d_code)
		self.codebooks = nn.ParameterList([
			nn.Parameter(torch.randn(k, d_code) * 0.1) for k in codebook_sizes
		])
		self.beta = beta

	def forward(self, x):
		# x: [B,T,D_in]
		z = self.proj_in(x)  # [B,T,D]
		residual = z
		deq = torch.zeros_like(residual)
		indices = []
		vq_loss = torch.tensor(0.0, device=x.device)
		for cb in self.codebooks:
			# [K,D]
			# compute distances: ||r - e||^2 = r^2 + e^2 -2 r.e
			r = residual.unsqueeze(-2)            # [B,T,1,D]
			e = cb.unsqueeze(0).unsqueeze(0)      # [1,1,K,D]
			dist = (r - e).pow(2).sum(-1)         # [B,T,K]
			idx = dist.argmin(dim=-1)             # [B,T]
			indices.append(idx)
			q = F.embedding(idx, cb)              # [B,T,D]
			deq = deq + q
			# straight-through
			vq_loss = vq_loss + self.beta * (residual.detach() - q).pow(2).mean() + (residual - q.detach()).pow(2).mean()
			residual = residual - q
		return deq, indices, vq_loss 