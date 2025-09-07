import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class PhaseAwareSpectralLoss(nn.Module):
	"""Enhanced spectral loss with phase coherence."""
	def __init__(self, fft_sizes=[512, 1024, 2048], gamma=0.5):
		super().__init__()
		self.fft_sizes = fft_sizes
		self.gamma = gamma

	def forward(self, pred, target):
		total_loss = 0
		for fft_size in self.fft_sizes:
			hop_size = fft_size // 4
			
			# Compute STFT
			pred_stft = torch.stft(pred, fft_size, hop_size, return_complex=True, normalized=True)
			target_stft = torch.stft(target, fft_size, hop_size, return_complex=True, normalized=True)
			
			# Magnitude loss with perceptual weighting
			pred_mag = torch.abs(pred_stft) ** self.gamma
			target_mag = torch.abs(target_stft) ** self.gamma
			mag_loss = F.l1_loss(pred_mag, target_mag)
			
			# Phase consistency loss
			pred_phase = torch.angle(pred_stft)
			target_phase = torch.angle(target_stft)
			# Unwrap phase differences and compute circular distance
			phase_diff = pred_phase - target_phase
			phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
			phase_loss = torch.mean(torch.abs(phase_diff))
			
			# Combine with frequency weighting (emphasize lower frequencies)
			freq_weights = torch.exp(-torch.arange(fft_size // 2 + 1, device=pred.device).float() / (fft_size // 8))
			mag_loss = mag_loss * freq_weights.unsqueeze(0).unsqueeze(0)
			phase_loss = phase_loss * freq_weights.unsqueeze(0).unsqueeze(0)
			
			total_loss += mag_loss + 0.1 * phase_loss
		
		return total_loss / len(self.fft_sizes)


class PitchConsistencyLoss(nn.Module):
	"""Loss to enforce pitch periodicity."""
	def __init__(self, sample_rate=16000):
		super().__init__()
		self.sample_rate = sample_rate

	def forward(self, pred, periods):
		# pred: [B, T], periods: [B, F] (frame-level periods)
		B, T = pred.shape
		frame_size = T // periods.size(1)
		
		total_loss = 0
		count = 0
		
		for b in range(B):
			for f in range(periods.size(1)):
				start_idx = f * frame_size
				end_idx = min((f + 1) * frame_size, T)
				
				if end_idx - start_idx < 40:  # Skip short segments
					continue
					
				segment = pred[b, start_idx:end_idx]
				period = int(periods[b, f].item())
				
				if period < 32 or period > 255:  # Skip invalid periods
					continue
				
				# Compute autocorrelation around the expected period
				if len(segment) > period + 10:
					seg_len = len(segment) - period
					sig1 = segment[:seg_len]
					sig2 = segment[period:period + seg_len]
					
					# Normalized cross-correlation
					corr = F.cosine_similarity(sig1.unsqueeze(0), sig2.unsqueeze(0), dim=1)
					pitch_loss = 1.0 - corr.mean()
					
					total_loss += pitch_loss
					count += 1
		
		return total_loss / max(count, 1)


class PerceptualLoss(nn.Module):
	"""Mel-scale perceptual loss with temporal dynamics."""
	def __init__(self, sample_rate=16000, n_mels=80, f_min=80, f_max=7600):
		super().__init__()
		self.mel_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=sample_rate, n_fft=1024, hop_length=256, 
			n_mels=n_mels, f_min=f_min, f_max=f_max
		)

	def forward(self, pred, target):
		# Compute mel spectrograms
		pred_mel = torch.log(self.mel_transform(pred) + 1e-8)
		target_mel = torch.log(self.mel_transform(target) + 1e-8)
		
		# Static loss
		static_loss = F.l1_loss(pred_mel, target_mel)
		
		# Dynamic loss (temporal derivatives)
		pred_delta = pred_mel[:, :, 1:] - pred_mel[:, :, :-1]
		target_delta = target_mel[:, :, 1:] - target_mel[:, :, :-1]
		dynamic_loss = F.l1_loss(pred_delta, target_delta)
		
		return static_loss + 0.5 * dynamic_loss


class AdversarialFeatureMatchingLoss(nn.Module):
	"""Enhanced feature matching with adaptive weighting."""
	def __init__(self):
		super().__init__()

	def forward(self, real_features, fake_features):
		loss = 0
		total_weight = 0
		
		for real_feats, fake_feats in zip(real_features, fake_features):
			for i, (real_feat, fake_feat) in enumerate(zip(real_feats[:-1], fake_feats[:-1])):
				# Adaptive weighting: deeper layers get more weight
				weight = 2 ** i
				layer_loss = F.l1_loss(fake_feat, real_feat.detach())
				loss += weight * layer_loss
				total_weight += weight
		
		return loss / total_weight


class EnhancedCompositeLoss(nn.Module):
	"""Composite loss combining all components with adaptive weighting."""
	def __init__(self, sample_rate=16000):
		super().__init__()
		self.spectral_loss = PhaseAwareSpectralLoss()
		self.pitch_loss = PitchConsistencyLoss(sample_rate)
		self.perceptual_loss = PerceptualLoss(sample_rate)
		self.feature_loss = AdversarialFeatureMatchingLoss()
		
		# Learnable loss weights
		self.weight_spectral = nn.Parameter(torch.tensor(1.0))
		self.weight_pitch = nn.Parameter(torch.tensor(0.1))
		self.weight_perceptual = nn.Parameter(torch.tensor(0.5))
		self.weight_feature = nn.Parameter(torch.tensor(1.0))
		self.weight_adversarial = nn.Parameter(torch.tensor(1.0))

	def forward(self, pred, target, periods=None, real_features=None, fake_features=None, 
				fake_scores=None, mode='generator'):
		losses = {}
		total_loss = 0
		
		# Core reconstruction losses
		spectral = self.spectral_loss(pred, target)
		perceptual = self.perceptual_loss(pred, target)
		
		losses['spectral'] = spectral
		losses['perceptual'] = perceptual
		
		total_loss += F.softplus(self.weight_spectral) * spectral
		total_loss += F.softplus(self.weight_perceptual) * perceptual
		
		# Pitch consistency (if periods provided)
		if periods is not None:
			pitch = self.pitch_loss(pred, periods)
			losses['pitch'] = pitch
			total_loss += F.softplus(self.weight_pitch) * pitch
		
		# Adversarial losses (if in generator mode)
		if mode == 'generator':
			if fake_scores is not None:
				adv_loss = 0
				for scores in fake_scores:
					adv_loss += ((1 - scores[-1]) ** 2).mean()
				adv_loss /= len(fake_scores)
				losses['adversarial'] = adv_loss
				total_loss += F.softplus(self.weight_adversarial) * adv_loss
			
			if real_features is not None and fake_features is not None:
				feat_loss = self.feature_loss(real_features, fake_features)
				losses['feature_matching'] = feat_loss
				total_loss += F.softplus(self.weight_feature) * feat_loss
		
		losses['total'] = total_loss
		return total_loss, losses 