import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GeneralizedSSM(nn.Module):
    """
    基于MambaJSCC论文的广义状态空间模型(GSSM)
    融合到FarGanRVQ现有框架中
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 线性投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM 参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        
        # 可学习参数 A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        
        # 可学习参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # CSI-ReST 参数
        self.csi_inject_interval = 4  # CSI注入间隔
        
    def forward(self, x, csi=None):
        """
        Args:
            x: [B, L, D] 输入序列
            csi: [B,] 信道状态信息 (SNR)
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # 分割成 x 和 z
        
        # 卷积处理
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[..., :L]  # 保持长度不变
        x = x.transpose(1, 2)  # [B, L, d_inner]
        
        # SiLU 激活
        x = F.silu(x)
        
        # 双向GSSM处理
        y1 = self._gssm_forward(x, scan_reverse=False, csi=csi)
        y2 = self._gssm_forward(x, scan_reverse=True, csi=csi)
        
        # 合并双向结果
        y = y1 + y2
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def _gssm_forward(self, x, scan_reverse=False, csi=None):
        """GSSM前向传播，实现CSI-ReST"""
        B, L, D = x.shape
        
        # 扫描变换
        if scan_reverse:
            x_scan = torch.flip(x, dims=[1])  # 反向扫描
        else:
            x_scan = x  # 正向扫描
        
        # 计算SSM参数
        dt = F.softplus(self.dt_proj(x_scan))  # [B, L, d_state]
        B_proj = self.x_proj(x_scan)  # [B, L, 2*d_state]
        B_mat, C = B_proj.chunk(2, dim=-1)  # [B, L, d_state] each
        
        A = -torch.exp(self.A_log.float())  # [d_state]
        
        # 离散化
        dtA = torch.einsum('bld,d->bld', dt, A)  # [B, L, d_state]
        dtB = torch.einsum('bld,bld->bld', dt, B_mat)  # [B, L, d_state]
        
        # 初始化隐状态
        batch_size = x_scan.shape[0]
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        
        # CSI-ReST: 将CSI注入初始状态
        if csi is not None:
            h[:, 0] = csi  # 将SNR注入第一个状态维度
        
        # 递归计算（预分配输出，避免列表拼接与小张量重复分配）
        y = x_scan.new_empty(B, L, D)
        exp_dtA = torch.exp(dtA)  # [B, L, d_state]
        for t in range(L):
            # 状态更新
            h = h * exp_dtA[:, t] + dtB[:, t] * x_scan[:, t, :self.d_state]
            
            # CSI-ReST: 定期重新注入CSI
            if csi is not None and t % self.csi_inject_interval == 0:
                h[:, 0] = csi
            
            # 输出计算（用expand避免repeat拷贝）
            # 标量相关性: [B]
            y_scalar = torch.einsum('bd,bd->b', h, C[:, t, :])
            # 写入到预分配张量: [B, D]
            y[:, t, :] = y_scalar.unsqueeze(-1).expand(-1, D)
        
        # 残差连接
        y = y + x_scan * self.D
        
        # 反向扫描恢复
        if scan_reverse:
            y = torch.flip(y, dims=[1])
        
        return y


class MambaConditionNet(nn.Module):
    """
    基于GSSM的条件网络，替换原有的EnhancedConditionNet
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Mamba层序列
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'gssm': GeneralizedSSM(hidden_dim, d_state=16),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            })
            for _ in range(n_layers)
        ])
        
        # 时域上采样 (10ms -> 2.5ms)
        self.temporal_upsample = nn.ConvTranspose1d(
            hidden_dim, out_dim, kernel_size=4, stride=4, padding=0
        )
        
        # 增益预测
        self.gain_head = nn.Linear(out_dim, 1)
        
        # CSI门控
        self.csi_gate = nn.Linear(1, hidden_dim)
        
    def forward(self, x, csi=None):
        """
        Args:
            x: [B, T, in_dim] 输入条件
            csi: [B,] 信道状态信息
        Returns:
            cond: [B, T*4, out_dim] 子帧级条件
            gain: [B, T*4, 1] 增益
        """
        # 输入投影
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Mamba层处理
        for layer in self.mamba_layers:
            # 层归一化
            residual = x
            x = layer['norm'](x)
            
            # GSSM处理，注入CSI
            gssm_out = layer['gssm'](x, csi=csi)
            
            # CSI门控调制
            if csi is not None:
                csi_gate = torch.sigmoid(self.csi_gate(csi.unsqueeze(-1)))  # [B, hidden_dim]
                csi_gate = csi_gate.unsqueeze(1)  # [B, 1, hidden_dim]
                # 扩展csi_gate到与gssm_out相同的维度
                csi_gate = csi_gate.expand(-1, gssm_out.shape[1], -1)  # [B, L, hidden_dim]
                gssm_out = gssm_out * csi_gate
            
            x = residual + gssm_out
            
            # MLP分支
            residual = x
            x = residual + layer['mlp'](layer['norm'](x))
        
        # 时域上采样到子帧级别
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        cond = self.temporal_upsample(x)  # [B, out_dim, T*4]
        cond = cond.transpose(1, 2)  # [B, T*4, out_dim]
        
        # 增益预测（上界保护，避免数值爆炸导致音频剪裁/NaN）
        gain_raw = self.gain_head(cond)
        gain = torch.exp(gain_raw.clamp(-1.5, 1.5))  # [B, T*4, 1]

        return cond, gain


class CSIAwareSubframeNet(nn.Module):
    """
    基于现有SubframeNet但增加CSI感知能力
    """
    def __init__(self, cond_dim=96, hidden_dim=256, subframe_size=40,
                 use_dither: bool = True, noise_amp: float = 1.0/127.0,
                 use_pitch_gate: bool = True):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.subframe_size = subframe_size
        self.use_dither = use_dither
        self.noise_amp = noise_amp
        self.use_pitch_gate = use_pitch_gate
        
        # 前一子帧和基音处理
        self.prev_proj = nn.Conv1d(subframe_size, 32, 1)
        self.pitch_proj = nn.Conv1d(subframe_size, 32, 1)
        
        # 主干网络
        total_dim = 32 + 32 + cond_dim  # prev + pitch + cond
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(total_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
        ])
        
        # GLU门控
        self.gate_layers = nn.ModuleList([
            nn.Conv1d(cond_dim, hidden_dim, 1) for _ in range(3)
        ])
        
        # CSI门控
        self.csi_gates = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(3)
        ])
        
        # 简化 pitch 门控：从条件预测逐层门控；并用1x1卷积将 pitch_proj 注入每层
        if self.use_pitch_gate:
            self.pitch_gate_head = nn.Conv1d(cond_dim, 3, 1)
            self.pitch_inj = nn.ModuleList([
                nn.Conv1d(32, hidden_dim, 1) for _ in range(3)
            ])
        else:
            self.pitch_gate_head = None
            self.pitch_inj = None

        # 输出层
        self.output_proj = nn.Conv1d(hidden_dim, subframe_size, 1)

    def _n(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dither:
            return x.clamp(-1.0, 1.0)
        return (x + self.noise_amp * (torch.rand_like(x) - 0.5)).clamp(-1.0, 1.0)
        
    def forward(self, cond_subframe, prev_subframe, pitch_pred, csi=None):
        """
        Args:
            cond_subframe: [B, cond_dim] 当前子帧条件
            prev_subframe: [B, subframe_size] 前一子帧
            pitch_pred: [B, subframe_size] 基音预测
            csi: [B,] 信道状态信息
        """
        # 投影（加入微抖动，软限幅）
        prev_in = self._n(prev_subframe.unsqueeze(-1))           # [B, 40, 1]
        pitch_in = self._n(pitch_pred.unsqueeze(-1))             # [B, 40, 1]
        cond_in = self._n(cond_subframe.unsqueeze(-1))           # [B, cond_dim, 1]

        prev_proj = self.prev_proj(prev_in)                      # [B, 32, 1]
        pitch_proj = self.pitch_proj(pitch_in)                   # [B, 32, 1]
        cond_proj = cond_in                                      # [B, cond_dim, 1]

        if self.use_pitch_gate:
            pg = torch.sigmoid(self.pitch_gate_head(cond_proj)).squeeze(-1)  # [B,3]

        # 特征拼接
        x = torch.cat([prev_proj, pitch_proj, cond_proj], dim=1)  # [B, total_dim, 1]

        # 主干网络处理
        for i, (conv, gate) in enumerate(zip(self.conv_layers, self.gate_layers)):
            residual = x if x.shape[1] == self.hidden_dim else None

            # 主分支
            h = torch.tanh(conv(x))                              # [B, hidden_dim, 1]

            # 注入简化的 pitch 贡献
            if self.use_pitch_gate:
                inj = self.pitch_inj[i](pitch_proj)              # [B, hidden_dim, 1]
                gain_i = pg[:, i].view(-1, 1, 1)                 # [B,1,1]
                h = h + gain_i * inj
            h = self._n(h)

            # 门控分支
            g = torch.sigmoid(gate(cond_proj))                   # [B, hidden_dim, 1]
            g = self._n(g)

            # CSI调制
            if csi is not None:
                csi_mod = torch.sigmoid(self.csi_gates[i](csi.unsqueeze(-1))).unsqueeze(-1)  # [B, hidden_dim, 1]
                g = g * csi_mod

            # GLU
            x = self._n(h * g)

            # 残差连接
            if residual is not None:
                x = x + residual

        # 输出投影
        output = self.output_proj(x).squeeze(-1)  # [B, subframe_size]
        
        return output

    def sequence_forward(self, cond_seq, prev_seq, pitch_seq, gain, csi=None):
        """
        训练期并行子帧（Teacher Forcing）前向：整段子帧一次性并行计算。
        Args:
            cond_seq:  [B, T_sub, cond_dim]
            prev_seq:  [B, T_sub, subframe_size]
            pitch_seq: [B, T_sub, subframe_size]
            gain:      [B, T_sub, 1]
            csi:       [B,] or None
        Returns:
            [B, T_sub*subframe_size]
        """
        B, T_sub, _ = cond_seq.shape

        # 转换为卷积格式并加入微抖动: [B, D, T]
        cond_t = self._n(cond_seq.transpose(1, 2))    # [B, cond_dim, T]
        prev_t = self._n(prev_seq.transpose(1, 2))    # [B, 40, T]
        pitch_t = self._n(pitch_seq.transpose(1, 2))  # [B, 40, T]

        # 投影和拼接
        prev_proj = self.prev_proj(prev_t)      # [B, 32, T]
        pitch_proj = self.pitch_proj(pitch_t)   # [B, 32, T]
        x = torch.cat([prev_proj, pitch_proj, cond_t], dim=1)  # [B, 32+32+cond_dim, T]

        # 主干网络 + 门控 + （可选）CSI调制
        # 逐层 pitch 门控（时间维度）
        if self.use_pitch_gate:
            pg_t = torch.sigmoid(self.pitch_gate_head(cond_t))   # [B,3,T]

        for i, (conv, gate) in enumerate(zip(self.conv_layers, self.gate_layers)):
            residual = x if x.shape[1] == self.hidden_dim else None

            h = torch.tanh(conv(x))                 # [B, hidden_dim, T]
            if self.use_pitch_gate:
                inj = self.pitch_inj[i](pitch_proj) # [B, hidden_dim, T]
                gain_i = pg_t[:, i:i+1, :]          # [B,1,T]
                h = h + gain_i * inj
            h = self._n(h)

            g = torch.sigmoid(gate(cond_t))         # [B, hidden_dim, T]
            g = self._n(g)

            if csi is not None:
                csi_mod = torch.sigmoid(self.csi_gates[i](csi.unsqueeze(-1))).unsqueeze(-1)  # [B, hidden_dim, 1]
                csi_mod = csi_mod.expand(-1, -1, h.shape[-1])  # [B, hidden_dim, T]
                g = g * csi_mod

            x = self._n(h * g)
            if residual is not None:
                x = x + residual

        # 输出与增益
        out = self.output_proj(x)                  # [B, 40, T]
        out = out.transpose(1, 2)                  # [B, T, 40]
        out = out * gain                           # [B, T, 40]

        return out.reshape(B, -1)                  # [B, T*40]


class MambaEnhancedFarGan(nn.Module):
    """
    基于现有EnhancedFarGan框架融合MambaJSCC技术
    """
    def __init__(self, in_features=20, cond_dim=32, subframe_size=40):
        super().__init__()
        self.subframe_size = subframe_size
        
        # 输入处理（保持与原框架一致）
        self.feature_proj = nn.Linear(in_features, cond_dim)
        self.pitch_embed = nn.Embedding(224, 8)  # period embedding
        self.voicing_proj = nn.Linear(1, 8)
        
        # 核心组件（用Mamba技术替换）
        self.cond_net = MambaConditionNet(cond_dim + 16, 128, 96)  # +16 for pitch/voicing
        self.subframe_net = CSIAwareSubframeNet(96, 256, subframe_size)
        
        # 基音预测器（保持原有逻辑）
        self.pitch_predictor = AdaptivePitchPredictor(64)
        
        # 基音缓冲区
        self.register_buffer('pitch_buffer', torch.zeros(1, 256))
        
        # JSCC编解码（可选）
        self.jscc_encoder = nn.Sequential(
            nn.Conv1d(96, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 24, 3, padding=1),
        )
        
        self.jscc_decoder = nn.Sequential(
            nn.Conv1d(24, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 96, 3, padding=1),
        )
        
        # CSI自适应层
        self.csi_adapt = nn.Linear(1, 96)
        
    def forward(self, features, periods=None, voicing=None, csi=None, 
                channel_noise=None, target_length=None,
                parallel_train: bool = False, teacher_signal: torch.Tensor = None):
        """
        Args:
            features: [B, T10, F] 输入特征
            periods: [B, T10] 基音周期
            voicing: [B, T10] 清浊音
            csi: [B,] 信道状态信息 (SNR)
            channel_noise: 信道噪声仿真
            target_length: 目标输出长度
        """
        B, T10, F = features.shape
        device = features.device
        
        # 处理缺失的基音/清浊音信息
        if periods is None:
            periods = torch.full((B, T10), 100.0, device=device)
        if voicing is None:
            voicing = torch.ones(B, T10, device=device)
        
        # 特征处理（与原版 FARGAN 时间对齐）：
        # 原版在 cond_net 中丢弃前2帧（features[:,2:], periods[:,2:])，
        # 且在逐帧合成时使用 period[:, 3+n]。这里复现同样的时序偏移。
        feats_shift = features[:, 2:, :] if features.shape[1] > 2 else features
        periods_shift = periods[:, 2:] if periods is not None and periods.shape[1] > 2 else periods

        feat_proj = self.feature_proj(feats_shift)
        pitch_emb = self.pitch_embed(((periods_shift if periods_shift is not None else periods) - 32).long().clamp(0, 223)) if periods is not None else torch.zeros(feat_proj.shape[0], feat_proj.shape[1], 8, device=device, dtype=feat_proj.dtype)
        voicing_emb = self.voicing_proj(voicing.unsqueeze(-1))
        
        # 组合特征
        combined_feat = torch.cat([feat_proj, pitch_emb, voicing_emb], dim=-1)
        
        # Mamba条件网络处理
        cond_subframe, gain = self.cond_net(combined_feat, csi=csi)
        
        # 可选JSCC编解码
        if channel_noise is not None:
            # JSCC编码
            encoded = self.jscc_encoder(cond_subframe.transpose(1, 2))
            
            # 信道仿真
            if isinstance(channel_noise, str) and channel_noise == 'auto':
                if csi is not None:
                    snr_linear = 10 ** (csi / 10.0)  # [B]
                    # 计算每个样本的信号功率
                    signal_power = torch.var(encoded, dim=(1, 2), keepdim=True)  # [B, 1, 1]
                    noise_power = signal_power / snr_linear.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                    noise = torch.sqrt(noise_power) * torch.randn_like(encoded)
                else:
                    # 默认SNR = 10dB
                    signal_power = torch.var(encoded)
                    noise_power = signal_power / 10.0
                    noise = torch.sqrt(noise_power) * torch.randn_like(encoded)
                received = encoded + noise
            else:
                received = encoded + channel_noise
            
            # JSCC解码
            decoded = self.jscc_decoder(received)
            cond_subframe = decoded.transpose(1, 2)
            
            # CSI自适应调制
            if csi is not None:
                csi_weight = torch.tanh(self.csi_adapt(csi.unsqueeze(-1)))  # [B, 96]
                csi_weight = csi_weight.unsqueeze(1)  # [B, 1, 96] for broadcasting
                cond_subframe = cond_subframe * (1 + 0.1 * csi_weight)
        
        # 目标长度调整
        if target_length is not None:
            target_subframes = target_length // self.subframe_size
            if target_subframes < cond_subframe.shape[1]:
                cond_subframe = cond_subframe[:, :target_subframes]
                gain = gain[:, :target_subframes]
        
        # 子帧合成：支持训练期并行Teacher Forcing
        T_sub = cond_subframe.shape[1]
        # 允许在 eval() 下也使用并行 Teacher Forcing（只依赖参数而非 self.training）
        if parallel_train and (teacher_signal is not None):
            # 构造教师驱动的 prev/pitch 序列
            total_needed = T_sub * self.subframe_size
            teacher_trim = teacher_signal[:, :total_needed]
            tgt_subframes = teacher_trim.reshape(B, T_sub, self.subframe_size)

            prev_seq = torch.zeros_like(tgt_subframes)
            if T_sub > 1:
                prev_seq[:, 1:, :] = tgt_subframes[:, :-1, :]
            pitch_seq = prev_seq  # 最少侵入：用 prev 近似 pitch

            output = self.subframe_net.sequence_forward(
                cond_seq=cond_subframe,
                prev_seq=prev_seq,
                pitch_seq=pitch_seq,
                gain=gain,
                csi=csi,
            )
        else:
            # 逐子帧自回归（增强版：激励记忆 + 周期索引 + 写回）
            outputs = []
            # 激励记忆（长度256），用于按周期抽取pitch excitation，并写回滑窗
            exc_mem = torch.zeros(B, 256, device=device)

            # 可选：从教师信号进行预热（若提供 teacher_signal 则将首帧写入 exc_mem 尾部）
            if teacher_signal is not None and teacher_signal.shape[1] >= self.subframe_size * 4:
                # 取一帧(160)进行记忆预热；更长也只需最后一帧即可提供相位锚点
                exc_mem[:, -160:] = teacher_signal[:, :160]

            # 子帧递推
            subframe_size = self.subframe_size
            for t in range(T_sub):
                curr_cond = cond_subframe[:, t]     # [B, 96]
                curr_gain = gain[:, t]              # [B, 1]

                # 从激励记忆取 prev（上一子帧）
                prev_subframe = exc_mem[:, -subframe_size:]  # [B, 40]

                # 基于周期的激励抽取（与原FARGAN一致的索引策略）
                # period 映射到当前子帧所属帧
                if periods is None:
                    # 缺省周期：用中值100
                    period_t = torch.full((B,), 100.0, device=device)
                else:
                    # 对齐原版：period 索引使用 3 + frame_idx
                    frame_idx = t // 4
                    per_idx = min(3 + frame_idx, periods.shape[1] - 1)
                    period_t = periods[:, per_idx]  # [B]

                # idx = 256 - period + (arange(44) - 2); 超界回绕 period
                rng = torch.arange(subframe_size + 4, device=device)
                idx = (256 - period_t.long().clamp(32, 255)).unsqueeze(1) + rng.unsqueeze(0) - 2
                mask = idx >= 256
                idx = idx - mask * period_t.long().unsqueeze(1)
                pitch_window = torch.gather(exc_mem, 1, idx)  # [B, 44]
                pitch_pred = pitch_window[:, 2:-2]            # [B, 40]

                # 归一到增益尺度，避免在子网内部过饱和（与原版思路一致）
                inv_gain = (1.0 / (1e-5 + curr_gain.squeeze(-1))).unsqueeze(1)  # [B,1]
                prev_scaled = prev_subframe * inv_gain
                pitch_scaled = pitch_pred * inv_gain

                # 子帧合成
                subframe_out = self.subframe_net(
                    curr_cond, prev_scaled, pitch_scaled, csi=csi
                )

                # 增益应用
                subframe_out = subframe_out * curr_gain

                # 写回激励记忆（滑窗）
                exc_mem = torch.cat([exc_mem[:, subframe_size:], subframe_out.detach()], dim=1)

                outputs.append(subframe_out)

            output = torch.cat(outputs, dim=1)
        
        return output
    
    def _simple_pitch_prediction(self, prev_subframe, period):
        """简化的基音预测"""
        # 基于周期的历史查找
        T = period.long().clamp(32, 255)
        pitch_pred = torch.zeros_like(prev_subframe)
        
        for i, t in enumerate(T):
            if t <= prev_subframe.shape[1]:
                pitch_pred[i, :t] = prev_subframe[i, -t:]
            else:
                # 周期大于子帧长度时的处理
                pitch_pred[i] = prev_subframe[i]
        
        return pitch_pred


class AdaptivePitchPredictor(nn.Module):
    """保持原有的自适应基音预测器"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 多抽头权重学习
        self.tap_weights = nn.Parameter(torch.tensor([0.7, 0.2, 0.1]))
        
        # Voicing感知层
        self.voicing_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        
        # 时频自适应卷积
        self.adaptive_conv = nn.Conv1d(1, 1, 3, padding=1)
        
    def forward(self, prev_samples, period, voicing):
        """自适应基音预测"""
        # 多抽头插值
        taps = []
        for i, weight in enumerate(self.tap_weights):
            offset = period + i
            if offset < prev_samples.shape[1]:
                tap = prev_samples[:, -offset:]
                taps.append(weight * tap)
        
        if taps:
            pitch_pred = sum(taps) / len(taps)
        else:
            pitch_pred = prev_samples
        
        # Voicing门控
        voicing_gate = self.voicing_gate(voicing.unsqueeze(-1))
        pitch_pred = pitch_pred * voicing_gate.squeeze(-1)
        
        # 时频自适应
        pitch_pred = self.adaptive_conv(pitch_pred.unsqueeze(1)).squeeze(1)
        
        return pitch_pred


# 保持与现有损失函数的兼容性
class MambaJSCCEnhancedLoss(nn.Module):
    """增强的损失函数，支持CSI自适应 + 短窗时域/幅度/直流约束"""
    def __init__(self, spectral_weight=1.0, adversarial_weight=0.1,
                 channel_weight=0.05, phase_weight=0.1,
                 sig_weight: float = 0.0, rms_weight: float = 0.0, dc_weight: float = 0.0,
                 sig_win: int = 80, sig_hop: int = 80,
                 debug: bool = False, stft_sizes=None):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.adversarial_weight = adversarial_weight
        self.channel_weight = channel_weight
        self.phase_weight = phase_weight
        self.sig_weight = sig_weight
        self.rms_weight = rms_weight
        self.dc_weight = dc_weight
        self.sig_win = sig_win
        self.sig_hop = sig_hop
        self.debug = debug
        
        # 多分辨率STFT损失
        if stft_sizes is None:
            stft_sizes = [512, 1024, 2048]
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size=size, hop_size=size//4)
            for size in stft_sizes
        ])
        
    def forward(self, pred, target, csi=None, disc_real=None, disc_fake=None,
                pred_raw: torch.Tensor = None, target_raw: torch.Tensor = None):
        losses = {}
        
        # 基础光谱损失
        spectral_loss = 0
        for stft_loss in self.stft_losses:
            spectral_loss += stft_loss(pred, target)
        losses['spectral'] = spectral_loss * self.spectral_weight
        
        # 对抗损失
        if disc_real is not None and disc_fake is not None:
            adv_loss = 0
            # disc_fake是判别器列表，每个判别器返回特征列表
            for i, disc_scores in enumerate(disc_fake):
                try:
                    # 取最后一层的输出作为判别结果
                    if isinstance(disc_scores, (list, tuple)):
                        disc_output = disc_scores[-1]
                    else:
                        disc_output = disc_scores
                    
                    # 确保disc_output是tensor
                    if isinstance(disc_output, (list, tuple)):
                        disc_output = disc_output[-1] if len(disc_output) > 0 else disc_output[0]
                    
                    if isinstance(disc_output, torch.Tensor):
                        adv_loss += F.mse_loss(disc_output, torch.ones_like(disc_output))
                    else:
                        if self.debug:
                            print(f"警告: 判别器{i}输出类型异常: {type(disc_output)}")
                        
                except Exception as e:
                    if self.debug:
                        print(f"警告: 判别器{i}对抗损失计算失败: {e}")
                        print(f"  disc_scores类型: {type(disc_scores)}")
                        if isinstance(disc_scores, (list, tuple)):
                            print(f"  disc_scores长度: {len(disc_scores)}")
                            for j, item in enumerate(disc_scores):
                                print(f"    [{j}]: {type(item)}")
                    continue
                    
            if len(disc_fake) > 0:
                adv_loss = adv_loss / len(disc_fake)  # 平均损失
                losses['adversarial'] = adv_loss * self.adversarial_weight
        
        # CSI自适应损失
        if csi is not None:
            snr_weight = torch.exp(-csi / 10.0)  # [B]
            snr_weight = snr_weight.unsqueeze(-1)  # [B, 1] for broadcasting
            l1_loss = F.l1_loss(pred, target, reduction='none')  # [B, N]
            channel_loss = torch.mean(snr_weight * l1_loss)
            losses['channel'] = channel_loss * self.channel_weight
        
        # 相位感知损失
        phase_loss = self._phase_loss(pred, target)
        losses['phase'] = phase_loss * self.phase_weight

        # 短窗时域相位/瞬态约束 + RMS/直流（在原始波形上计算更有效）
        if (self.sig_weight > 0 or self.rms_weight > 0 or self.dc_weight > 0):
            x = pred_raw if pred_raw is not None else pred
            y = target_raw if target_raw is not None else target
            s_loss, r_loss, d_loss = self._short_time_losses(x, y, self.sig_win, self.sig_hop)
            if self.sig_weight > 0:
                losses['sig'] = s_loss * self.sig_weight
            if self.rms_weight > 0:
                losses['rms'] = r_loss * self.rms_weight
            if self.dc_weight > 0:
                losses['dc'] = d_loss * self.dc_weight
        
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return losses
    
    def _phase_loss(self, pred, target):
        """相位感知损失"""
        # 缓存窗口到buffer，避免每步重建
        if not hasattr(self, '_phase_window'):
            self.register_buffer('_phase_window', torch.hann_window(1024), persistent=False)
        window = self._phase_window.to(device=pred.device, dtype=pred.dtype)
        pred_stft = torch.stft(pred, n_fft=1024, hop_length=256, window=window, return_complex=True)
        target_stft = torch.stft(target, n_fft=1024, hop_length=256, window=window, return_complex=True)
        
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        
        # 相位差异，避免跳跃
        phase_diff = pred_phase - target_phase
        phase_loss = F.l1_loss(torch.sin(phase_diff), torch.zeros_like(phase_diff))
        
        return phase_loss

    def _short_time_losses(self, pred, target, win: int = 80, hop: int = 80):
        """计算短窗损失：
        - sig_loss: 单位能量余弦距离（原 FARGAN 思路）
        - rms_loss: 每窗 log-RMS 的 L1
        - dc_loss: 每窗均值的 L2
        """
        B, T = pred.shape[0], pred.shape[1]
        # 使用 unfold 进行分帧: [B, nwin, win]
        if T < win:
            # 退化：用整段作为一窗
            x = pred.unsqueeze(1)
            y = target.unsqueeze(1)
        else:
            x = pred.unfold(dimension=1, size=win, step=hop)
            y = target.unfold(dimension=1, size=win, step=hop)
        nwin = x.shape[1]
        eps = 1e-8

        # sig_loss: 单位能量归一化后余弦距离
        xn = x / (x.norm(dim=-1, keepdim=True) + eps)
        yn = y / (y.norm(dim=-1, keepdim=True) + eps)
        cos_sim = (xn * yn).sum(dim=-1)
        sig_loss = (1.0 - cos_sim).mean()

        # rms 匹配：log-RMS 的 L1
        xr = torch.sqrt((x.pow(2).mean(dim=-1)) + eps)
        yr = torch.sqrt((y.pow(2).mean(dim=-1)) + eps)
        rms_loss = torch.abs(torch.log(xr + eps) - torch.log(yr + eps)).mean()

        # dc：均值的 L2
        xdc = x.mean(dim=-1)
        dc_loss = xdc.pow(2).mean()

        return sig_loss, rms_loss, dc_loss


class STFTLoss(nn.Module):
    """STFT光谱损失"""
    def __init__(self, fft_size=1024, hop_size=256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        # 预先在CPU上生成窗口，forward时移动到目标设备
        self.register_buffer('_win_cpu', torch.hann_window(self.fft_size), persistent=False)
        
    def forward(self, pred, target):
        window = self._win_cpu.to(device=pred.device, dtype=pred.dtype)
        pred_stft = torch.stft(pred, n_fft=self.fft_size, hop_length=self.hop_size, window=window, return_complex=True)
        target_stft = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, window=window, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        return mag_loss 
