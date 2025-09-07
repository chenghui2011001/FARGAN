import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptivePitchPredictor(nn.Module):
	"""Enhanced pitch prediction with learned interpolation and voicing-aware gating."""
	def __init__(self, hidden_dim=64):
		super().__init__()
		self.pitch_dense = nn.Linear(1, hidden_dim)
		self.voicing_dense = nn.Linear(1, hidden_dim)
		self.adapt_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
		self.gate_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)
		self.interp_weights = nn.Parameter(torch.tensor([0.7, 0.2, 0.1]))

	def forward(self, pitch_buf, period, voicing, subframe_idx):
		# pitch_buf: [B, 256+], period: [B], voicing: [B], subframe_idx: int
		device = pitch_buf.device
		B = pitch_buf.size(0)
		
		# Multi-tap pitch prediction with learned interpolation
		period_int = period.long().clamp(32, 255)
		taps = []
		for i, w in enumerate(self.interp_weights):
			offset = period_int + i * 8  # staggered taps
			offset = offset.clamp(0, pitch_buf.size(1) - 40)
			idx = offset[:, None] + torch.arange(40, device=device)[None, :]
			tap = torch.gather(pitch_buf, 1, idx)
			taps.append(w * tap)
		
		pred = sum(taps)  # [B, 40]
		
		# Voicing-aware adaptive gating
		v_feat = self.voicing_dense(voicing[:, None])  # [B, H]
		p_feat = self.pitch_dense(period[:, None] / 255.0)  # [B, H]
		combined = v_feat + p_feat  # [B, H]
		
		# Temporal adaptation
		combined = combined[:, :, None].expand(-1, -1, 40)  # [B, H, 40]
		adapted = torch.tanh(self.adapt_conv(combined))
		gate = torch.sigmoid(self.gate_conv(adapted)).squeeze(1)  # [B, 40]
		
		return pred * gate


class MambaBlock(nn.Module):
	"""Simplified Mamba-like block for efficient sequential modeling."""
	def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
		super().__init__()
		self.d_model = d_model
		self.d_state = d_state
		d_inner = int(expand * d_model)
		
		self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
		self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv-1, groups=d_inner)
		self.x_proj = nn.Linear(d_inner, d_state, bias=False)
		self.dt_proj = nn.Linear(d_inner, d_state, bias=True)
		self.out_proj = nn.Linear(d_inner, d_model, bias=False)
		
		# Initialize state transition
		self.A_log = nn.Parameter(torch.randn(d_state))
		self.D = nn.Parameter(torch.ones(d_inner))

	def forward(self, x):
		# x: [B, L, D]
		B, L, D = x.shape
		x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
		x, res = x_and_res.split(x_and_res.size(-1) // 2, dim=-1)
		
		# Conv path
		x = x.transpose(1, 2)  # [B, d_inner, L]
		x = self.conv1d(x)[:, :, :L]  # causal conv
		x = x.transpose(1, 2)  # [B, L, d_inner]
		
		# SSM path (simplified)
		x = F.silu(x)
		dt = F.softplus(self.dt_proj(x))  # [B, L, d_state]
		A = -torch.exp(self.A_log.float()).unsqueeze(0).unsqueeze(0)  # [1, 1, d_state]
		
		# Simplified state space (parallel scan approximation)
		x_proj = self.x_proj(x)  # [B, L, d_state]
		y = x_proj * torch.exp(A * dt)  # simplified computation
		y = y.sum(dim=-1, keepdim=True).expand(-1, -1, x.size(-1))  # [B, L, d_inner]
		
		# Combine with residual and output
		y = y * F.silu(res)
		y = self.out_proj(y)
		return y


class EnhancedConditionNet(nn.Module):
	"""Enhanced conditioning with better upsampling and normalization."""
	def __init__(self, in_dim=32, hidden_dim=128, out_dim=96):
		super().__init__()
		self.in_proj = nn.Linear(in_dim, hidden_dim)
		self.mamba1 = MambaBlock(hidden_dim)
		self.mamba2 = MambaBlock(hidden_dim)
		
		# Better upsampling with learned interpolation
		self.upsample_conv = nn.ConvTranspose1d(hidden_dim, out_dim, kernel_size=8, stride=4, padding=2)
		self.refine_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
		
		# Adaptive normalization
		self.gain_proj = nn.Linear(out_dim, 1)
		self.bias_proj = nn.Linear(out_dim, 1)

	def forward(self, x):
		# x: [B, T10, F]
		x = F.gelu(self.in_proj(x))  # [B, T10, H]
		x = self.mamba1(x)
		x = self.mamba2(x)
		
		# Upsample to subframe rate
		x = x.transpose(1, 2)  # [B, H, T10]
		x = self.upsample_conv(x)  # [B, Out, T40]
		x = F.gelu(self.refine_conv(x))
		x = x.transpose(1, 2)  # [B, T40, Out]
		
		# Adaptive gain/bias
		gain = torch.exp(self.gain_proj(x))  # [B, T40, 1]
		bias = self.bias_proj(x)  # [B, T40, 1]
		
		return x, gain, bias


class EnhancedSubframeNet(nn.Module):
	"""Enhanced subframe synthesis with better pitch integration."""
	def __init__(self, cond_dim=96, hidden_dim=256, subframe_size=40):
		super().__init__()
		self.subframe_size = subframe_size
		
		# Input projections
		self.cond_proj = nn.Linear(cond_dim, hidden_dim)
		self.prev_conv = nn.Conv1d(subframe_size, 64, kernel_size=3, padding=1)
		self.pitch_conv = nn.Conv1d(subframe_size, 64, kernel_size=3, padding=1)
		
		# Enhanced processing blocks
		self.mamba1 = MambaBlock(hidden_dim + 128)  # 64+64 from prev/pitch
		self.mamba2 = MambaBlock(hidden_dim + 128)
		self.mamba3 = MambaBlock(hidden_dim + 128)
		
		# Output with residual connection
		self.out_proj = nn.Linear(hidden_dim + 128, subframe_size)
		self.residual_gate = nn.Linear(hidden_dim + 128, 1)

	def forward(self, cond, prev_sf, pitch_sf, gain, bias):
		# cond: [B, T40, C], others: [B, T40, 40], gain/bias: [B, T40, 1]
		B, T, _ = cond.shape
		
		# Process inputs
		c = self.cond_proj(cond)  # [B, T40, H]
		
		# Convolution on prev/pitch (treat as sequences)
		prev_flat = prev_sf.reshape(B * T, self.subframe_size).unsqueeze(1)  # [BT, 1, 40]
		pitch_flat = pitch_sf.reshape(B * T, self.subframe_size).unsqueeze(1)
		
		prev_feat = self.prev_conv(prev_flat).squeeze(1).reshape(B, T, 64)  # [B, T, 64]
		pitch_feat = self.pitch_conv(pitch_flat).squeeze(1).reshape(B, T, 64)
		
		# Combine features
		x = torch.cat([c, prev_feat, pitch_feat], dim=-1)  # [B, T, H+128]
		
		# Sequential processing
		x = self.mamba1(x)
		x = self.mamba2(x)
		x = self.mamba3(x)
		
		# Output with adaptive normalization
		out = self.out_proj(x)  # [B, T, 40]
		
		# Residual gating for stability
		res_gate = torch.sigmoid(self.residual_gate(x))
		out = out + res_gate * prev_sf  # residual connection
		
		# Apply adaptive gain/bias
		out = out * gain + bias
		
		# Flatten to time-domain signal
		return out.reshape(B, -1)


class EnhancedFarGan(nn.Module):
	"""FarGan with key architectural improvements and optimizations."""
	def __init__(self, in_features=20, cond_dim=32, subframe_size=40):
		super().__init__()
		self.subframe_size = subframe_size
		
		# Input processing
		self.feature_proj = nn.Linear(in_features, cond_dim)
		self.pitch_embed = nn.Embedding(224, 8)  # period embedding
		self.voicing_proj = nn.Linear(1, 8)
		
		# Core components
		self.cond_net = EnhancedConditionNet(cond_dim + 16, 128, 96)  # +16 for pitch/voicing
		self.pitch_predictor = AdaptivePitchPredictor(64)
		self.subframe_net = EnhancedSubframeNet(96, 256, subframe_size)
		
		# Pitch buffer
		self.register_buffer('pitch_buffer', torch.zeros(1, 256))

	def forward(self, features, periods=None, voicing=None):
		# features: [B, T10, F], periods: [B, T10], voicing: [B, T10]
		B, T10, F = features.shape
		device = features.device
		
		# Handle missing pitch/voicing (use dummy values for MVP)
		if periods is None:
			periods = torch.full((B, T10), 100.0, device=device)
		if voicing is None:
			voicing = torch.ones(B, T10, device=device)
		
		# Feature processing
		feat_proj = self.feature_proj(features)
		pitch_emb = self.pitch_embed((periods - 32).long().clamp(0, 223))
		voicing_emb = self.voicing_proj(voicing.unsqueeze(-1))
		
		# Combine features
		combined_feat = torch.cat([feat_proj, pitch_emb, voicing_emb], dim=-1)
		
		# Conditioning
		cond, gain, bias = self.cond_net(combined_feat)  # [B, T40, 96]
		
		# Initialize pitch buffer for batch
		if self.pitch_buffer.size(0) != B:
			self.pitch_buffer = torch.zeros(B, 256, device=device)
		
		# Generate subframes
		T40 = cond.size(1)
		prev_sf = torch.zeros(B, T40, self.subframe_size, device=device)
		pitch_sf = torch.zeros(B, T40, self.subframe_size, device=device)
		
		# Simple pitch prediction (can be enhanced with actual buffer management)
		for i in range(T40):
			frame_idx = i // 4
			if frame_idx < periods.size(1):
				period = periods[:, frame_idx]
				voice = voicing[:, frame_idx]
				pitch_sf[:, i, :] = self.pitch_predictor(
					self.pitch_buffer, period, voice, i % 4
				)
		
		# Synthesize
		output = self.subframe_net(cond, prev_sf, pitch_sf, gain, bias)
		
		# Update pitch buffer (simplified)
		if output.size(1) >= self.subframe_size:
			self.pitch_buffer = torch.cat([
				self.pitch_buffer[:, output.size(1):],
				output[:, -self.subframe_size:]
			], dim=1)
		
		return output 