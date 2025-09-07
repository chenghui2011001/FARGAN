import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GeneralizedSSM(nn.Module):
    """
    基于MambaJSCC论文的广义状态空间模型(GSSM)
    支持任意扫描方案和CSI-ReST信道自适应
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, scan_scheme='bidirectional'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.scan_scheme = scan_scheme
        
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
        
        # 可学习参数 A (N x N)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(1, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # 可学习参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 扫描变换矩阵
        self.register_buffer('R_kappa1', torch.eye(d_model))  # 正向扫描
        self.register_buffer('R_kappa2', self._create_reverse_scan_matrix(d_model))  # 反向扫描
        
        # CSI-ReST 参数
        self.csi_inject_interval = 4  # CSI注入间隔
        
    def _create_reverse_scan_matrix(self, size):
        """创建反向扫描矩阵"""
        R = torch.zeros(size, size)
        for i in range(size):
            R[i, size - 1 - i] = 1.0
        return R
    
    def forward(self, x, csi=None):
        """
        Args:
            x: [B, L, D] 输入序列
            csi: [B,] 信道状态信息 (SNR)
        Returns:
            y: [B, L, D] 输出序列
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
        y1 = self._gssm_forward(x, self.R_kappa1, csi)
        y2 = self._gssm_forward(x, self.R_kappa2, csi)
        
        # 合并双向结果
        y = y1 + y2
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def _gssm_forward(self, x, R_matrix, csi):
        """
        GSSM前向传播，实现CSI-ReST
        """
        B, L, D = x.shape
        
        # 扫描变换
        x_flat = x.view(B, -1)  # [B, L*D]
        x_scan = torch.matmul(x_flat, R_matrix.T)  # 扫描变换
        x_scan = x_scan.view(B, L, D)
        
        # 计算SSM参数
        dt = F.softplus(self.dt_proj(x_scan))  # [B, L, d_state]
        B_proj = self.x_proj(x_scan)  # [B, L, 2*d_state]
        B, C = B_proj.chunk(2, dim=-1)  # [B, L, d_state] each
        
        A = -torch.exp(self.A_log.float())  # [d_state]
        
        # 离散化
        dtA = torch.einsum('bld,d->bld', dt, A)  # [B, L, d_state]
        dtB = torch.einsum('bld,bld->bld', dt, B)  # [B, L, d_state]
        
        # 初始化隐状态
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        
        # CSI-ReST: 将CSI注入初始状态
        if csi is not None:
            h[:, 0] = csi  # 将SNR注入第一个状态维度
        
        # 递归计算
        outputs = []
        for t in range(L):
            # 状态更新
            h = h * torch.exp(dtA[:, t]) + dtB[:, t] * x_scan[:, t, :self.d_state]
            
            # CSI-ReST: 定期重新注入CSI
            if csi is not None and t % self.csi_inject_interval == 0:
                h[:, 0] = csi
            
            # 输出计算
            y_t = torch.einsum('bd,bld->bl', h, C[:, t:t+1])  # [B, 1]
            y_t = y_t.unsqueeze(-1).repeat(1, 1, D)  # [B, 1, D]
            outputs.append(y_t)
        
        y = torch.cat(outputs, dim=1)  # [B, L, D]
        
        # 残差连接
        y = y + x_scan * self.D
        
        # 反向扫描恢复
        y_flat = y.view(B, -1)
        y_recover = torch.matmul(y_flat, R_matrix)  # 恢复变换
        y = y_recover.view(B, L, D)
        
        return y


class CSIAwareMambaBlock(nn.Module):
    """
    信道状态感知的Mamba块，融合CSI-ReST机制
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.gssm = GeneralizedSSM(d_model, d_state, d_conv, expand)
        
        # MLP分支
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # CSI门控
        self.csi_gate = nn.Linear(1, d_model)
        
    def forward(self, x, csi=None):
        """
        Args:
            x: [B, L, D] 输入
            csi: [B,] 信道状态信息
        """
        # GSSM分支
        residual = x
        x = self.norm1(x)
        gssm_out = self.gssm(x, csi)
        
        # CSI门控调制
        if csi is not None:
            csi_gate = torch.sigmoid(self.csi_gate(csi.unsqueeze(-1)))  # [B, 1, D]
            gssm_out = gssm_out * csi_gate
        
        x = residual + gssm_out
        
        # MLP分支
        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        
        return x


class MambaFarGanJSCC(nn.Module):
    """
    融合MambaJSCC技术的FarGan模型
    支持GSSM、CSI-ReST和信道自适应编解码
    """
    def __init__(self, in_features=20, cond_dim=128, n_mamba_layers=4):
        super().__init__()
        self.in_features = in_features
        self.cond_dim = cond_dim
        self.n_mamba_layers = n_mamba_layers
        
        # 条件网络 - 使用Mamba架构
        self.cond_proj = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        
        # Mamba条件处理层
        self.mamba_layers = nn.ModuleList([
            CSIAwareMambaBlock(d_model=64, d_state=16)
            for _ in range(n_mamba_layers)
        ])
        
        # 时域上采样到子帧级别
        self.temporal_upsample = nn.ConvTranspose1d(64, cond_dim, kernel_size=4, stride=4)
        
        # 增益预测
        self.gain_proj = nn.Linear(cond_dim, 1)
        
        # 子帧合成网络
        self.subframe_net = EnhancedSubframeNet(cond_dim)
        
        # JSCC编码器
        self.jscc_encoder = JSCCEncoder(cond_dim)
        
        # JSCC解码器
        self.jscc_decoder = JSCCDecoder(cond_dim)
        
    def forward(self, features, target_length=None, csi=None, channel_noise=None):
        """
        Args:
            features: [B, T, F] 输入特征
            target_length: 目标输出长度
            csi: [B,] 信道状态信息 (SNR)
            channel_noise: 信道噪声仿真
        """
        B, T, F = features.shape
        
        # 条件投影
        cond = self.cond_proj(features)  # [B, T, 64]
        
        # Mamba层处理，注入CSI
        for layer in self.mamba_layers:
            cond = layer(cond, csi)
        
        # 如果有信道仿真，通过JSCC编解码
        if channel_noise is not None:
            # JSCC编码
            encoded = self.jscc_encoder(cond)
            
            # 信道传输仿真
            received = self._channel_simulation(encoded, channel_noise, csi)
            
            # JSCC解码
            cond = self.jscc_decoder(received, csi)
        
        # 时域上采样
        cond = cond.transpose(1, 2)  # [B, 64, T]
        cond_subframe = self.temporal_upsample(cond)  # [B, cond_dim, T*4]
        cond_subframe = cond_subframe.transpose(1, 2)  # [B, T*4, cond_dim]
        
        # 增益计算
        gain = torch.exp(self.gain_proj(cond_subframe))  # [B, T*4, 1]
        
        # 目标长度调整
        if target_length is not None:
            target_subframes = target_length // 40
            if target_subframes < cond_subframe.shape[1]:
                cond_subframe = cond_subframe[:, :target_subframes]
                gain = gain[:, :target_subframes]
        
        # 子帧合成
        output = self.subframe_net(cond_subframe, gain, csi)
        
        return output
    
    def _channel_simulation(self, x, noise, csi):
        """信道传输仿真"""
        if noise is None:
            return x
        
        # 添加高斯噪声
        if isinstance(noise, torch.Tensor):
            x_noisy = x + noise
        else:
            # 根据SNR添加噪声
            snr_linear = 10 ** (csi / 10.0) if csi is not None else 10.0
            noise_power = torch.var(x) / snr_linear
            noise_tensor = torch.sqrt(noise_power) * torch.randn_like(x)
            x_noisy = x + noise_tensor
        
        return x_noisy


class JSCCEncoder(nn.Module):
    """JSCC编码器，实现信源信道联合编码"""
    def __init__(self, d_model):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model // 8, 3, padding=1),
        )
        
        # 错误保护编码
        self.protection = nn.Conv1d(d_model // 8, d_model // 4, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D] 输入特征
        Returns:
            encoded: [B, T, D//4] 编码后特征
        """
        x = x.transpose(1, 2)  # [B, D, T]
        compressed = self.compress(x)  # [B, D//8, T]
        protected = self.protection(compressed)  # [B, D//4, T]
        return protected.transpose(1, 2)  # [B, T, D//4]


class JSCCDecoder(nn.Module):
    """JSCC解码器，实现信道自适应解码"""
    def __init__(self, d_model):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv1d(d_model // 4, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
        )
        
        # CSI自适应层
        self.csi_adapt = nn.Linear(1, d_model)
        
    def forward(self, x, csi=None):
        """
        Args:
            x: [B, T, D//4] 接收特征
            csi: [B,] 信道状态信息
        Returns:
            decoded: [B, T, D] 解码后特征
        """
        x = x.transpose(1, 2)  # [B, D//4, T]
        decoded = self.expand(x)  # [B, D, T]
        decoded = decoded.transpose(1, 2)  # [B, T, D]
        
        # CSI自适应调制
        if csi is not None:
            csi_weight = torch.tanh(self.csi_adapt(csi.unsqueeze(-1)))  # [B, 1, D]
            decoded = decoded * (1 + 0.1 * csi_weight)
        
        return decoded


class EnhancedSubframeNet(nn.Module):
    """增强的子帧合成网络，融合CSI感知"""
    def __init__(self, cond_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        
        # 前一子帧处理
        self.prev_proj = nn.Conv1d(40, 32, 1)
        
        # 基音预测处理
        self.pitch_proj = nn.Conv1d(40, 32, 1)
        
        # 主干网络
        total_dim = 32 + 32 + cond_dim  # prev + pitch + cond
        
        self.main_net = nn.ModuleList([
            nn.Conv1d(total_dim, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
        ])
        
        # GLU门控
        self.gates = nn.ModuleList([
            nn.Conv1d(cond_dim, 256, 1) for _ in range(3)
        ])
        
        # CSI门控
        self.csi_gate = nn.Linear(1, 256)
        
        # 输出层
        self.output_proj = nn.Conv1d(256, 40, 1)
        
    def forward(self, cond_subframe, gain, csi=None):
        """
        Args:
            cond_subframe: [B, T_sub, cond_dim] 条件特征
            gain: [B, T_sub, 1] 增益
            csi: [B,] 信道状态信息
        """
        B, T_sub, _ = cond_subframe.shape
        
        # 初始化输出
        outputs = []
        prev_subframe = torch.zeros(B, 40, device=cond_subframe.device)
        
        for t in range(T_sub):
            # 当前条件
            curr_cond = cond_subframe[:, t]  # [B, cond_dim]
            curr_gain = gain[:, t]  # [B, 1]
            
            # 基音预测（简化版）
            pitch_pred = self._simple_pitch_prediction(prev_subframe)
            
            # 前一子帧投影
            prev_proj = self.prev_proj(prev_subframe.unsqueeze(-1)).squeeze(-1)  # [B, 32]
            
            # 基音投影
            pitch_proj = self.pitch_proj(pitch_pred.unsqueeze(-1)).squeeze(-1)  # [B, 32]
            
            # 特征拼接
            x = torch.cat([
                prev_proj.unsqueeze(-1),
                pitch_proj.unsqueeze(-1),
                curr_cond.unsqueeze(-1).repeat(1, 1, 1)
            ], dim=1)  # [B, total_dim, 1]
            
            # 主干网络处理
            for i, (conv, gate_conv) in enumerate(zip(self.main_net, self.gates)):
                residual = x if x.shape[1] == 256 else None
                
                # 主分支
                h = torch.tanh(conv(x))
                
                # 门控分支
                gate = torch.sigmoid(gate_conv(curr_cond.unsqueeze(-1)))
                
                # CSI调制
                if csi is not None:
                    csi_mod = torch.sigmoid(self.csi_gate(csi.unsqueeze(-1))).unsqueeze(-1)  # [B, 256, 1]
                    gate = gate * csi_mod
                
                # GLU
                x = h * gate
                
                # 残差连接
                if residual is not None:
                    x = x + residual
            
            # 输出投影
            subframe_out = self.output_proj(x).squeeze(-1)  # [B, 40]
            
            # 增益应用
            subframe_out = subframe_out * curr_gain.squeeze(-1)
            
            # 更新前一子帧
            prev_subframe = subframe_out.detach()
            
            outputs.append(subframe_out)
        
        # 拼接所有子帧
        output = torch.cat(outputs, dim=1)  # [B, T_sub * 40]
        
        return output
    
    def _simple_pitch_prediction(self, prev_subframe):
        """简化的基音预测"""
        # 这里用简单的历史重复作为基音预测
        # 实际实现中应该根据基音周期进行更精确的预测
        return prev_subframe.clone()


class MambaJSCCLoss(nn.Module):
    """
    MambaJSCC损失函数，包含光谱损失、对抗损失和信道自适应损失
    """
    def __init__(self, 
                 spectral_weight=1.0,
                 adversarial_weight=0.1,
                 channel_weight=0.05):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.adversarial_weight = adversarial_weight
        self.channel_weight = channel_weight
        
        # 多分辨率STFT损失
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size=size, hop_size=size//4)
            for size in [512, 1024, 2048]
        ])
        
    def forward(self, pred, target, csi=None, disc_real=None, disc_fake=None):
        """
        Args:
            pred: [B, N] 预测信号
            target: [B, N] 目标信号
            csi: [B,] 信道状态信息
            disc_real/disc_fake: 判别器输出
        """
        losses = {}
        
        # 光谱损失
        spectral_loss = 0
        for stft_loss in self.stft_losses:
            spectral_loss += stft_loss(pred, target)
        losses['spectral'] = spectral_loss * self.spectral_weight
        
        # 对抗损失
        if disc_real is not None and disc_fake is not None:
            adv_loss = F.mse_loss(disc_fake[-1], torch.ones_like(disc_fake[-1]))
            losses['adversarial'] = adv_loss * self.adversarial_weight
        
        # 信道自适应损失
        if csi is not None:
            # 低SNR时增加光谱损失权重
            snr_weight = torch.exp(-csi / 10.0)  # 低SNR时权重更高
            channel_loss = torch.mean(snr_weight * F.l1_loss(pred, target, reduction='none'))
            losses['channel'] = channel_loss * self.channel_weight
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


class STFTLoss(nn.Module):
    """STFT光谱损失"""
    def __init__(self, fft_size=1024, hop_size=256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        
    def forward(self, pred, target):
        # 计算STFT
        pred_stft = torch.stft(pred, n_fft=self.fft_size, hop_length=self.hop_size, 
                              return_complex=True)
        target_stft = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size,
                                return_complex=True)
        
        # 幅度损失
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        # 相位损失
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.l1_loss(torch.sin(pred_phase), torch.sin(target_phase))
        
        return mag_loss + 0.1 * phase_loss 