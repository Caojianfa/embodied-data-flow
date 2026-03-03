"""视觉-语言多模态融合模块

架构：Cross-Attention + LayerNorm + FFN（残差连接）

语义：视觉特征作为 Query，语言特征作为 Key/Value。
     "用语言指令引导视觉特征关注语义相关区域。"

输入：
  visual:   (B, dim) 或 (dim,)   — 视觉特征
  language: (dim,)               — 语言特征（整个 batch 共用同一条指令）

输出：
  fused: 与 visual 形状相同
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionLanguageFusion(nn.Module):
    """
    视觉-语言 Cross-Attention 融合模块

    Args:
        dim:       特征维度，与 CLIP 输出维度一致（默认 512）
        num_heads: 多头注意力头数（默认 8）
        dropout:   Attention dropout 比率（默认 0.0）
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        # Cross-Attention：视觉为 Q，语言为 K/V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        # FFN：两层全连接，GELU 激活，4× 扩展
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual:   视觉特征，shape=(B, dim) 或 (dim,)
            language: 语言特征，shape=(dim,)

        Returns:
            融合特征，shape 与 visual 相同
        """
        squeeze = visual.dim() == 1
        if squeeze:
            visual = visual.unsqueeze(0)  # → (1, dim)

        B = visual.shape[0]

        # language 扩展为 (B, 1, dim) 作为 K/V
        lang_kv = language.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        vis_q = visual.unsqueeze(1)  # (B, 1, dim)

        # Cross-Attention + 残差 + LayerNorm
        attn_out, _ = self.cross_attn(query=vis_q, key=lang_kv, value=lang_kv)
        x = self.norm1(visual + attn_out.squeeze(1))

        # FFN + 残差 + LayerNorm
        x = self.norm2(x + self.ffn(x))

        if squeeze:
            x = x.squeeze(0)
        return x
