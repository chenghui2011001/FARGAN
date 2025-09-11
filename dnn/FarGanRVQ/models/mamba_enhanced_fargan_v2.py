# dnn/FarGanRVQ/models/mamba_enhanced_fargan.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

__all__ = [
    "GeneralizedSSM",
    "MambaConditionNet",
    "CSIAwareSubframeNet",
    "MambaEnhancedFarGan",
    "AdaptivePitchPredictor",
    "MambaJSCCEnhancedLoss",
    "STFTLoss",
]

# =========================
# Generalized SSM (GSSM)
# =========================
class GeneralizedSSM(nn.Module):
    """
    广义 SSM：双向扫描 + 向量输出 + 残差 CSI 调制（不向 h 注入标量 CSI）
    - 输入: x[B,L,D]
    - 输出: y[B,L,D]
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # 输入投影 + 深度可分离 1D 卷积（非因果，保长度）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True
        )

        # SSM 参数头
        self.x_proj  = nn.Linear(self.d_inner, self.d_state * 2, bias=False)  # -> B_mat, (原 C_dyn 去掉)
        self.dt_proj = nn.Linear(self.d_inner, self.d_state,  bias=True)      # -> dt
        self.u_proj  = nn.Linear(self.d_inner, self.d_state,  bias=False)     # -> u_t

        # 连续系统 A<0（可学习）
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))                                # exp 再取负
        self.C     = nn.Linear(self.d_state, self.d_inner, bias=False)         # 向量读出
        self.D     = nn.Parameter(torch.ones(self.d_inner))                    # 残差通道缩放

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 轻量 CSI 残差调制（仅调制 D，不写入 h）
        self.csi_affine = nn.Sequential(nn.Linear(1, self.d_inner), nn.Tanh())

    def forward(self, x: torch.Tensor, csi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:   [B, L, d_model]
            csi: [B] or None（单位 dB）
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)                                  # [B,L,2H]
        x_h, z = xz.chunk(2, dim=-1)                          # [B,L,H], [B,L,H]

        x_h = self.conv1d(x_h.transpose(1, 2)).transpose(1, 2)
        x_h = F.silu(x_h)

        y_f = self._scan(x_h, reverse=False, csi=csi)         # [B,L,H]
        y_b = self._scan(x_h, reverse=True,  csi=csi)         # [B,L,H]
        y   = (y_f + y_b) * F.silu(z)                         # 门控融合
        return self.out_proj(y)                               # [B,L,D]

    def _scan(self, x_scan: torch.Tensor, reverse: bool, csi: Optional[torch.Tensor]) -> torch.Tensor:
        if reverse:
            x_scan = torch.flip(x_scan, dims=[1])             # [B,L,H]
        B, L, H = x_scan.shape
        S = self.d_state

        # 设备/精度对齐
        dev, dtype = x_scan.device, x_scan.dtype

        dt = F.softplus(self.dt_proj(x_scan))                 # [B,L,S]
        BC = self.x_proj(x_scan)                              # [B,L,2S]
        B_mat, _ = BC.chunk(2, dim=-1)                        # [B,L,S]

        A = -torch.exp(self.A_log.to(dtype))                  # [S]
        dtA = torch.einsum("bld,d->bld", dt, A)               # [B,L,S]
        exp_dtA = torch.exp(dtA)
        dtB = dt * B_mat                                      # [B,L,S]

        # CSI 残差通道调制: D_eff[B,H]
        if csi is not None:
            csi = csi.to(dtype=dtype, device=dev)
            m   = self.csi_affine(csi.unsqueeze(-1))          # [B,H]
            D_eff = self.D.to(device=dev, dtype=dtype).unsqueeze(0) * (1.0 + 0.1 * m)
        else:
            D_eff = self.D.to(device=dev, dtype=dtype).unsqueeze(0)

        # 递归状态与输出
        h = x_scan.new_zeros(B, S, dtype=dtype)
        y = x_scan.new_empty(B, L, H, dtype=dtype)
        C_lin = self.C.to(device=dev, dtype=dtype)

        for t in range(L):
            u_t = self.u_proj(x_scan[:, t, :])                # [B,S]
            h   = h * exp_dtA[:, t, :] + dtB[:, t, :] * u_t   # [B,S]
            y[:, t, :] = C_lin(h)                             # [B,H]

        y = y + x_scan * D_eff.unsqueeze(1)                   # 残差
        if reverse:
            y = torch.flip(y, dims=[1])
        return y


# =========================
# Condition Net (GSSM + MLP)
# =========================
class MambaConditionNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden_dim),
                "gssm" : GeneralizedSSM(hidden_dim, d_state=16),
                "norm2": nn.LayerNorm(hidden_dim),
                "mlp"  : nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            }) for _ in range(n_layers)
        ])
        self.temporal_upsample = nn.ConvTranspose1d(hidden_dim, out_dim, kernel_size=4, stride=4, padding=0)
        self.gain_head = nn.Linear(out_dim, 1)
        self.csi_gate  = nn.Linear(1, hidden_dim)

    def forward(self, x: torch.Tensor, csi: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T10, in_dim]
        Return:
          cond: [B, Tsub, out_dim]
          gain: [B, Tsub, 1] （受限增益）
        """
        x = self.input_proj(x)                                # [B,T,H]
        for lyr in self.layers:
            g = lyr["gssm"](lyr["norm1"](x), csi=csi)         # [B,T,H]
            if csi is not None:
                cgate = torch.sigmoid(self.csi_gate(csi.unsqueeze(-1)))  # [B,H]
                g = g * cgate.unsqueeze(1)                    # [B,T,H]
            x = x + g
            x = x + lyr["mlp"](lyr["norm2"](x))

        x    = x.transpose(1, 2)                              # [B,H,T]
        cond = self.temporal_upsample(x).transpose(1, 2)      # [B,T*4,out_dim]
        s    = self.gain_head(cond)                           # [B,T*4,1]
        gain = 1.0 + 0.5 * torch.tanh(s)                      # 受限增益，避免暴涨
        return cond, gain


# =========================
# Subframe Net (+ CSI gates)
# =========================
class CSIAwareSubframeNet(nn.Module):
    def __init__(self, cond_dim: int = 96, hidden_dim: int = 256, subframe_size: int = 40):
        super().__init__()
        self.subframe_size = subframe_size
        self.prev_proj  = nn.Conv1d(subframe_size, 32, 1)
        self.pitch_proj = nn.Conv1d(subframe_size, 32, 1)

        total = 32 + 32 + cond_dim
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(total,     hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        ])
        self.gate_layers = nn.ModuleList([nn.Conv1d(cond_dim, hidden_dim, 1) for _ in range(3)])
        self.csi_gates   = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(3)])
        self.output_proj = nn.Conv1d(hidden_dim, subframe_size, 1)

    def forward(
        self,
        cond_subframe: torch.Tensor,    # [B,cond]
        prev_subframe: torch.Tensor,    # [B,40]
        pitch_pred: torch.Tensor,       # [B,40]
        csi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        prev = self.prev_proj(prev_subframe.unsqueeze(-1))     # [B,32,1]
        pit  = self.pitch_proj(pitch_pred.unsqueeze(-1))       # [B,32,1]
        cnd  = cond_subframe.unsqueeze(-1)                     # [B,cond,1]
        x    = torch.cat([prev, pit, cnd], dim=1)              # [B,32+32+cond,1]

        for i, (conv, gate) in enumerate(zip(self.conv_layers, self.gate_layers)):
            res = x if x.shape[1] == conv.out_channels else None
            h = torch.tanh(conv(x))                            # [B,H,1]
            g = torch.sigmoid(gate(cnd))                       # [B,H,1]
            if csi is not None:
                cmod = torch.sigmoid(self.csi_gates[i](csi.unsqueeze(-1))).unsqueeze(-1)  # [B,H,1]
                g = g * cmod
            x = h * g
            if res is not None:
                x = x + res

        out = self.output_proj(x).squeeze(-1)                  # [B,40]
        return out

    def sequence_forward(
        self,
        cond_seq: torch.Tensor,        # [B,Tsub,cond]
        prev_seq: torch.Tensor,        # [B,Tsub,40]
        pitch_seq: torch.Tensor,       # [B,Tsub,40]
        gain: torch.Tensor,            # [B,Tsub,1]
        csi: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, Tsub, _ = cond_seq.shape
        cond_t  = cond_seq.transpose(1, 2)                     # [B,cond,T]
        prev_t  = prev_seq.transpose(1, 2)                     # [B,40,T]
        pitch_t = pitch_seq.transpose(1, 2)                    # [B,40,T]

        prev = self.prev_proj(prev_t)                          # [B,32,T]
        pit  = self.pitch_proj(pitch_t)                        # [B,32,T]
        x    = torch.cat([prev, pit, cond_t], dim=1)           # [B,32+32+cond,T]

        for i, (conv, gate) in enumerate(zip(self.conv_layers, self.gate_layers)):
            res = x if x.shape[1] == conv.out_channels else None
            h = torch.tanh(conv(x))                            # [B,H,T]
            g = torch.sigmoid(gate(cond_t))                    # [B,H,T]
            if csi is not None:
                cmod = torch.sigmoid(self.csi_gates[i](csi.unsqueeze(-1))).unsqueeze(-1)  # [B,H,1]
                g = g * cmod.expand(-1, -1, h.shape[-1])
            x = h * g
            if res is not None:
                x = x + res

        out = self.output_proj(x).transpose(1, 2)              # [B,T,40]
        out = out * gain                                       # [B,T,40]
        return out.reshape(B, -1)                              # [B,T*40]


# =========================
# 顶层模型
# =========================
class MambaEnhancedFarGan(nn.Module):
    """
    与现有训练/评估脚本接口兼容：
      forward(features, periods=None, voicing=None, csi=None,
              channel_noise=None, target_length=None,
              parallel_train=False, teacher_signal=None) -> [B,T]
    """
    def __init__(self, in_features: int = 20, cond_dim: int = 32, subframe_size: int = 40):
        super().__init__()
        self.subframe_size = subframe_size

        # 10ms 特征 → cond_dim
        self.feature_proj = nn.Linear(in_features, cond_dim)
        # pitch/voicing 融合
        self.pitch_embed  = nn.Embedding(224, 8)
        self.voicing_proj = nn.Linear(1, 8)

        # Mamba 条件 + 子帧网络
        self.cond_net     = MambaConditionNet(cond_dim + 16, 128, 96)
        self.subframe_net = CSIAwareSubframeNet(96, 256, subframe_size)

        # 可选：更复杂 pitch 预测器（默认不启用）
        self.pitch_predictor = AdaptivePitchPredictor(64)

        # JSCC（可选）
        self.jscc_encoder = nn.Sequential(
            nn.Conv1d(96, 48, 3, padding=1), nn.GELU(),
            nn.Conv1d(48, 24, 3, padding=1)
        )
        self.jscc_decoder = nn.Sequential(
            nn.Conv1d(24, 48, 3, padding=1), nn.GELU(),
            nn.Conv1d(48, 96, 3, padding=1)
        )
        self.csi_adapt = nn.Linear(1, 96)

    def forward(
        self,
        features: torch.Tensor,                    # [B,T10,F]
        periods: Optional[torch.Tensor] = None,    # [B,T10]
        voicing: Optional[torch.Tensor] = None,    # [B,T10]
        csi: Optional[torch.Tensor] = None,        # [B]
        channel_noise: Optional[torch.Tensor | str] = None,
        target_length: Optional[int] = None,
        parallel_train: bool = False,
        teacher_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T10, F = features.shape
        dev, dtype = features.device, features.dtype

        if periods is None:
            periods = torch.full((B, T10), 100.0, device=dev, dtype=dtype)
        if voicing is None:
            voicing = torch.ones(B, T10, device=dev, dtype=dtype)

        feat_proj   = self.feature_proj(features)                                 # [B,T10,cond]
        pitch_idx   = (periods - 32).clamp(0, 223).to(torch.long)                 # 安全索引
        pitch_emb   = self.pitch_embed(pitch_idx)                                 # [B,T10,8]
        voicing_emb = self.voicing_proj(voicing.unsqueeze(-1))                    # [B,T10,8]
        cond_in     = torch.cat([feat_proj, pitch_emb, voicing_emb], dim=-1)      # [B,T10,cond+16]

        cond_sf, gain = self.cond_net(cond_in, csi=csi)                           # [B,Tsub,96], [B,Tsub,1]

        # 可选 JSCC（对 Tsub 维度做编码/信道/解码）
        if channel_noise is not None:
            enc = self.jscc_encoder(cond_sf.transpose(1, 2))                      # [B,24,Tsub]
            if isinstance(channel_noise, str) and channel_noise == "auto":
                if csi is not None:
                    snr_lin = (10.0 ** (csi.to(dtype) / 10.0)).view(-1, 1, 1)
                    sig_pow = enc.var(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                    noise   = torch.randn_like(enc) * (sig_pow / snr_lin).sqrt()
                else:
                    sig_pow = enc.var().clamp_min(1e-8)
                    noise   = torch.randn_like(enc) * (sig_pow / 10.0).sqrt()     # 缺省 10 dB
                rx = enc + noise
            else:
                rx = enc + channel_noise
            dec = self.jscc_decoder(rx)                                           # [B,96,Tsub]
            cond_sf = dec.transpose(1, 2)                                         # [B,Tsub,96]
            if csi is not None:
                w = torch.tanh(self.csi_adapt(csi.unsqueeze(-1))).unsqueeze(1)    # [B,1,96]
                cond_sf = cond_sf * (1 + 0.1 * w)

        # 目标长度裁剪（与数据对齐）
        if target_length is not None:
            Tsub_need = int(target_length // self.subframe_size)
            if Tsub_need < cond_sf.shape[1]:
                cond_sf = cond_sf[:, :Tsub_need]
                gain    = gain[:, :Tsub_need]

        Tsub = cond_sf.shape[1]

        # 训练期并行 TF
        if self.training and parallel_train and (teacher_signal is not None):
            total_needed = Tsub * self.subframe_size
            tgt_trim = teacher_signal[:, :total_needed]
            tgt_sf   = tgt_trim.reshape(B, Tsub, self.subframe_size)

            prev_seq = torch.zeros_like(tgt_sf)
            if Tsub > 1:
                prev_seq[:, 1:, :] = tgt_sf[:, :-1, :]
            pitch_seq = prev_seq  # 最少侵入

            y = self.subframe_net.sequence_forward(cond_sf, prev_seq, pitch_seq, gain, csi)  # [B,T]
        else:
            # 自回归
            outs: List[torch.Tensor] = []
            prev_sf = torch.zeros(B, self.subframe_size, device=dev, dtype=dtype)
            for t in range(Tsub):
                cond_t = cond_sf[:, t]                                            # [B,96]
                gain_t = gain[:, t]                                               # [B,1]
                # 简化 pitch 预测：按 4 子帧 ≈ 1 帧
                idx = min(t // 4, T10 - 1)
                period_t = periods[:, idx]
                pitch_pred = self._simple_pitch_prediction(prev_sf, period_t)     # [B,40]
                sf = self.subframe_net(cond_t, prev_sf, pitch_pred, csi=csi) * gain_t
                prev_sf = sf.detach()
                outs.append(sf)
            y = torch.cat(outs, dim=1)                                           # [B,T]
        return y

    @staticmethod
    def _simple_pitch_prediction(prev_subframe: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        """
        prev_subframe: [B,40], period: [B]（单位：samples）
        """
        B, W = prev_subframe.shape
        pitch_pred = torch.zeros_like(prev_subframe)
        period_i = period.to(dtype=torch.long)
        for i in range(B):
            t = int(period_i[i].item())
            t = max(1, min(t, W))
            if t <= W:
                pitch_pred[i, :t] = prev_subframe[i, -t:]
            else:
                pitch_pred[i] = prev_subframe[i]
        return pitch_pred


# =========================
# （可选）更复杂 pitch 预测器
# =========================
class AdaptivePitchPredictor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.tap_weights = nn.Parameter(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32))
        self.voicing_gate = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())
        self.adaptive_conv = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, prev_samples: torch.Tensor, period: torch.Tensor, voicing: torch.Tensor) -> torch.Tensor:
        B, Tw = prev_samples.shape
        out = torch.zeros_like(prev_samples)
        for b in range(B):
            taps = []
            p0 = int(period[b].item())
            for i, w in enumerate(self.tap_weights):
                off = p0 + i
                if off < Tw:
                    taps.append(w * prev_samples[b, -off:])
            if taps:
                L = min(t.shape[0] for t in taps)
                acc = sum(t[-L:] for t in taps) / len(taps)
                out[b, -L:] = acc
            else:
                out[b] = prev_samples[b]
        vg = self.voicing_gate(voicing.view(-1, 1)).squeeze(-1)  # [B,hidden]
        scale = vg.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)    # 简化为标量门控
        out = out * scale
        out = self.adaptive_conv(out.unsqueeze(1)).squeeze(1)
        return out


# =========================
# Losses
# =========================
class MambaJSCCEnhancedLoss(nn.Module):
    """
    训练时常用：仅光谱 +（可选）CSI/L1 +（可选）相位
    对抗分支默认关闭（预训练/抛光期间不建议开启）
    """
    def __init__(
        self,
        spectral_weight: float = 1.0,
        adversarial_weight: float = 0.0,
        channel_weight: float = 0.05,
        phase_weight: float = 0.0,
        debug: bool = False,
        stft_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.adversarial_weight = adversarial_weight
        self.channel_weight = channel_weight
        self.phase_weight = phase_weight
        self.debug = debug

        if stft_sizes is None:
            stft_sizes = [512, 1024, 2048]
        self.stft_losses = nn.ModuleList([STFTLoss(fft_size=s, hop_size=s // 4) for s in stft_sizes])

    def forward(
        self,
        pred: torch.Tensor,            # [B,T]
        target: torch.Tensor,          # [B,T]
        csi: Optional[torch.Tensor] = None,
        disc_real=None, disc_fake=None
    ):
        losses = {}
        spec = 0.0
        for stft_loss in self.stft_losses:
            spec = spec + stft_loss(pred, target)
        losses["spectral"] = spec * self.spectral_weight

        if csi is not None and self.channel_weight > 0:
            snr_w = torch.exp(-csi.to(dtype=pred.dtype, device=pred.device) / 10.0).view(-1, 1)
            l1 = F.l1_loss(pred, target, reduction="none").mean(dim=1, keepdim=True)
            losses["channel"] = (snr_w * l1).mean() * self.channel_weight

        if self.phase_weight > 0:
            losses["phase"] = self._phase_loss(pred, target) * self.phase_weight

        total = 0.0
        for v in losses.values():
            total = total + v
        losses["total"] = total
        return losses

    def _phase_loss(self, pred: torch.Tensor, target: torch.Tensor):
        if not hasattr(self, "_phase_window"):
            self.register_buffer("_phase_window", torch.hann_window(1024), persistent=False)
        win = self._phase_window.to(device=pred.device, dtype=pred.dtype)
        X = torch.stft(pred, n_fft=1024, hop_length=256, window=win, return_complex=True)
        Y = torch.stft(target, n_fft=1024, hop_length=256, window=win, return_complex=True)
        dphi = torch.angle(X) - torch.angle(Y)
        return F.l1_loss(torch.sin(dphi), torch.zeros_like(dphi))


class STFTLoss(nn.Module):
    def __init__(self, fft_size: int = 1024, hop_size: int = 256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.register_buffer("_win_cpu", torch.hann_window(self.fft_size), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        win = self._win_cpu.to(device=pred.device, dtype=pred.dtype)
        X = torch.stft(pred, n_fft=self.fft_size, hop_length=self.hop_size, window=win, return_complex=True)
        Y = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, window=win, return_complex=True)
        return F.l1_loss(torch.abs(X), torch.abs(Y))
