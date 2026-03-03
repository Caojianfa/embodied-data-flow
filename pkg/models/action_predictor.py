"""Transformer 动作预测器

架构：Action Chunking with Transformers（ACT）的简化版
  - 一次性预测未来 T 步动作（Action Chunking），减少推理次数，预测更平滑
  - Transformer Encoder 做跨模态序列融合
  - 线性解码头输出连续动作

输入 Token 序列：
  Token 0:     视觉-语言融合特征（dim_fusion → d_model）
  Token 1..W:  IMU 窗口数据（imu_window 个采样，6维 → d_model）

输出：
  (B, action_horizon, action_dim)
  action[0:3] = delta_position (x, y, z)   末端执行器位移
  action[3:6] = delta_rotation (rx, ry, rz) 末端执行器旋转（轴角）
  action[6]   = gripper_openness ∈ [0, 1]   夹爪开合（Sigmoid 约束）

注意：本模型使用随机初始化权重做架构验证，
      动作标签来源于 EuRoC ground truth 速度（6D 伪标签）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionPredictor(nn.Module):
    """
    基于 Transformer Encoder 的动作预测器

    Args:
        fusion_dim:      视觉-语言融合特征维度（默认 512）
        imu_dim:         IMU 单步特征维度（默认 6：[wx,wy,wz,ax,ay,az]）
        imu_window:      IMU 历史窗口大小（默认 10）
        action_dim:      动作维度（默认 7）
        action_horizon:  预测未来步数（默认 10）
        d_model:         Transformer 内部维度（默认 256）
        nhead:           多头注意力头数（默认 8）
        num_layers:      Transformer Encoder 层数（默认 4）
        dropout:         Dropout 比率（默认 0.1）
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        imu_dim: int = 6,
        imu_window: int = 10,
        action_dim: int = 7,
        action_horizon: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.imu_window = imu_window
        self.d_model = d_model

        # 输入投影层
        self.fusion_proj = nn.Linear(fusion_dim, d_model)
        self.imu_proj = nn.Linear(imu_dim, d_model)

        # 可学习位置编码
        seq_len = 1 + imu_window  # 1 个视觉 token + W 个 IMU token
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 动作解码头
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim * action_horizon),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        fused_feat: torch.Tensor,
        imu_window: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_feat:  视觉-语言融合特征，shape=(B, fusion_dim)
            imu_window:  IMU 窗口，shape=(B, imu_window, 6)

        Returns:
            动作序列，shape=(B, action_horizon, action_dim)
        """
        B = fused_feat.shape[0]

        # 投影到统一维度
        vis_token = self.fusion_proj(fused_feat).unsqueeze(1)   # (B, 1, d_model)
        imu_tokens = self.imu_proj(imu_window)                  # (B, W, d_model)

        # 拼接 Token 序列并加位置编码
        tokens = torch.cat([vis_token, imu_tokens], dim=1)      # (B, 1+W, d_model)
        tokens = tokens + self.pos_embed

        # Transformer Encoder
        encoded = self.transformer(tokens)                       # (B, 1+W, d_model)

        # 取视觉 token（index=0）作为全局表示解码动作
        action_flat = self.action_head(encoded[:, 0, :])        # (B, T*7)
        actions = action_flat.view(B, self.action_horizon, self.action_dim)

        # 夹爪维度（index=6）约束到 [0,1]
        actions_out = actions.clone()
        actions_out[:, :, 6] = torch.sigmoid(actions[:, :, 6])

        return actions_out

    @torch.no_grad()
    def predict(
        self,
        fused_feat: torch.Tensor,
        imu_window: torch.Tensor,
    ) -> torch.Tensor:
        """推理接口（自动关闭梯度）"""
        self.eval()
        return self.forward(fused_feat, imu_window)
