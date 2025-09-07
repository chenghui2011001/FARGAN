import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeConditionNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 192):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.tconv = nn.ConvTranspose1d(hidden, out_dim, kernel_size=4, stride=4)
        self.gain = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T10, F]
        b, t, f = x.shape
        x = F.gelu(self.fc1(x))                # [B,T10,H]
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))                # [B,T10,H]
        x = x.transpose(1, 2)                  # [B,H,T10]
        x = F.gelu(self.conv1(x))              # [B,H,T10]
        x = F.gelu(self.conv2(x))              # [B,H,T10]
        x = self.tconv(x)                      # [B,Out,Tsub=T10*4]
        x = x.transpose(1, 2)                  # [B,Tsub,Out]
        gain = torch.exp(self.gain(x))         # [B,Tsub,1]
        return x, gain


class LargeGLUBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=5, padding=2)
        self.gate = nn.Conv1d(c_in, c_out, kernel_size=5, padding=2)
        self.norm = nn.GroupNorm(8, c_out)

    def forward(self, x):
        # x: [B, C, T]
        h = torch.tanh(self.conv(x))
        g = torch.sigmoid(self.gate(x))
        out = h * g
        return self.norm(out)


class LargeSubframeNet(nn.Module):
    def __init__(self, cond_dim: int = 192):
        super().__init__()
        self.pre_proj = nn.Conv1d(40, 64, kernel_size=1)
        self.pitch_proj = nn.Conv1d(40, 64, kernel_size=1)
        # cond_dim + 64 + 64 = cond_dim + 128
        self.in_proj = nn.Conv1d(cond_dim + 128, 512, kernel_size=3, padding=1)
        self.glu1 = LargeGLUBlock(512, 512)
        self.glu2 = LargeGLUBlock(512, 512)
        self.glu3 = LargeGLUBlock(512, 512)
        self.glu4 = LargeGLUBlock(512, 512)
        self.glu5 = LargeGLUBlock(512, 256)
        self.head = nn.Conv1d(256, 40, kernel_size=1)

    def forward(self, cond, prev, pitch, gain):
        # 所有输入: [B, Tsub, D]
        B, Tsub, _ = cond.shape
        
        # 转换为卷积格式: [B, D, Tsub]
        cond = cond.transpose(1, 2)     # [B, cond_dim, Tsub]
        prev = prev.transpose(1, 2)     # [B, 40, Tsub]
        pitch = pitch.transpose(1, 2)   # [B, 40, Tsub]
        gain = gain.transpose(1, 2)     # [B, 1, Tsub]
        
        # 投影和连接
        prev_proj = self.pre_proj(prev)      # [B, 64, Tsub]
        pitch_proj = self.pitch_proj(pitch)  # [B, 64, Tsub]
        
        x = torch.cat([cond, prev_proj, pitch_proj], dim=1)  # [B, cond_dim+128, Tsub]
        
        # 通过网络
        x = F.gelu(self.in_proj(x))          # [B, 512, Tsub]
        x = self.glu1(x)                     # [B, 512, Tsub]
        x = self.glu2(x)                     # [B, 512, Tsub]
        x = self.glu3(x)                     # [B, 512, Tsub]
        x = self.glu4(x)                     # [B, 512, Tsub]
        x = self.glu5(x)                     # [B, 256, Tsub]
        x = self.head(x)                     # [B, 40, Tsub]
        
        # 应用增益和重构
        x = x * gain                         # [B, 40, Tsub]
        x = x.transpose(1, 2)                # [B, Tsub, 40]
        x = x.reshape(B, Tsub * 40)          # [B, Tsub*40]
        
        return x


class LargerFarGan(nn.Module):
    """更大的 FarGan 模型以提高 GPU 利用率"""
    def __init__(self, in_features: int = 20, cond_dim: int = 192):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        self.cond_net = LargeConditionNet(32, hidden=256, out_dim=cond_dim)
        self.subframe = LargeSubframeNet(cond_dim)

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
        cond, gain = self.cond_net(c)                     # [B,Tsub,cond_dim], [B,Tsub,1]
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