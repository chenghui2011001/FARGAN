import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionNet(nn.Module):
	def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 96):
		super().__init__()
		self.fc = nn.Linear(in_dim, hidden)
		self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
		self.tconv = nn.ConvTranspose1d(hidden, out_dim, kernel_size=4, stride=4)
		self.gain = nn.Linear(out_dim, 1)

	def forward(self, x):
		# x: [B, T10, F]
		b, t, f = x.shape
		x = F.gelu(self.fc(x))                # [B,T10,H]
		x = x.transpose(1, 2)                 # [B,H,T10]
		x = F.gelu(self.conv(x))              # [B,H,T10]
		x = self.tconv(x)                     # [B,Out,Tsub=T10*4]
		x = x.transpose(1, 2)                 # [B,Tsub,Out]
		gain = torch.exp(self.gain(x))        # [B,Tsub,1]
		return x, gain


class GLUBlock(nn.Module):
	def __init__(self, c_in: int, c_out: int):
		super().__init__()
		self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
		self.gate = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)

	def forward(self, x):
		# x: [B, C, T]
		h = torch.tanh(self.conv(x))
		g = torch.sigmoid(self.gate(x))
		return h * g


class SubframeNet(nn.Module):
	def __init__(self, cond_dim: int = 96):
		super().__init__()
		self.pre_proj = nn.Conv1d(40, 32, kernel_size=1)
		self.pitch_proj = nn.Conv1d(40, 32, kernel_size=1)
		# cond_dim + 32 + 32 = cond_dim + 64
		self.in_proj = nn.Conv1d(cond_dim + 64, 160, kernel_size=3, padding=1)
		self.glu1 = GLUBlock(160, 160)
		self.glu2 = GLUBlock(160, 160)
		self.glu3 = GLUBlock(160, 160)
		self.head = nn.Conv1d(160, 40, kernel_size=1)

	def forward(self, cond, prev_sf, pitch_sf, gain):
		# cond: [B,Tsub,C], prev_sf: [B,Tsub,40], pitch_sf: [B,Tsub,40], gain: [B,Tsub,1]
		pc = self.pre_proj(prev_sf.transpose(1, 2))        # [B,32,T]
		pp = self.pitch_proj(pitch_sf.transpose(1, 2))     # [B,32,T]
		cc = cond.transpose(1, 2)                          # [B,C,T]
		x = torch.cat([cc, pc, pp], dim=1)                 # [B,C+64,T]
		x = self.in_proj(x)
		x = self.glu1(x)
		x = self.glu2(x)
		x = self.glu3(x)
		out = self.head(x)                                 # [B,40,T]
		out = out.transpose(1, 2)                          # [B,T,40]
		# apply gain
		out = out * gain
		# 重塑为最终波形：[B, T*40] 
		out = out.reshape(out.size(0), -1)
		return out


class OptimizedFarGan(nn.Module):
	"""Features -> condition network -> subframe synthesis -> waveform output.
	
	Input: [B, T_frames, F] where T_frames includes lookahead
	Output: [B, seq_len * 160] where seq_len is the target sequence length
	"""
	def __init__(self, in_features: int = 20, cond_dim: int = 96):
		super().__init__()
		self.cond_proj = nn.Linear(in_features, 32)
		self.cond_net = ConditionNet(32, 128, cond_dim)
		self.subframe = SubframeNet(cond_dim)

	def _build_pitch_from_prev(self, prev, Tsub):
		# naive placeholder: use prev as pitch buffer (no real shift)
		prev = prev.reshape(prev.size(0), Tsub, 40)
		pitch = prev.clone()
		return pitch

	def forward(self, features, target_length=None):
		"""
		Args:
			features: [B, T_frames, F] - input features with lookahead
			target_length: int - target output length in samples (optional)
		Returns:
			[B, N_samples] - synthesized waveform
		"""
		# features: [B,T_frames,F] -> project to condition space
		c = self.cond_proj(features)                       # [B,T_frames,32]
		cond, gain = self.cond_net(c)                     # [B,Tsub,96], [B,Tsub,1]
		B, Tsub, _ = cond.shape
		
		# 如果指定了目标长度，调整 Tsub
		if target_length is not None:
			target_subframes = target_length // 40
			if target_subframes < Tsub:
				cond = cond[:, :target_subframes, :]
				gain = gain[:, :target_subframes, :]
				Tsub = target_subframes
		
		# construct prev and pitch placeholders (zeros for first step)
		prev = torch.zeros(B, Tsub, 40, device=cond.device)
		pitch = self._build_pitch_from_prev(prev, Tsub)
		
		# 子帧合成
		y = self.subframe(cond, prev, pitch, gain)         # [B, Tsub*40]
		
		# 如果指定了目标长度，确保输出长度正确
		if target_length is not None and y.shape[1] != target_length:
			if y.shape[1] > target_length:
				y = y[:, :target_length]
			else:
				# 如果输出太短，用零填充
				padding = target_length - y.shape[1]
				y = F.pad(y, (0, padding), mode='constant', value=0)
		
		return y 