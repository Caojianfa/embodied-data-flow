"""CLIP ViT-B/32 视觉编码器封装

特点：
  - 惰性加载（首次调用时才下载/加载模型）
  - 支持单帧编码和批量编码
  - 同时支持文本编码（用于 VL 融合）
  - L2 归一化输出，特征在单位超球面上

CLIP 安装：
  pip install git+https://github.com/openai/CLIP.git
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from pkg.utils.logger import LoggerMixin


class VisionEncoder(LoggerMixin):
    """
    CLIP ViT-B/32 视觉编码器

    Args:
        model_name: CLIP 模型名，默认 "ViT-B/32"
        device:     推理设备，"cpu" 或 "cuda"
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._preprocess = None

    # ── 模型加载（惰性） ──────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import clip  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "请先安装 CLIP：\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )
        self.logger.info(
            "加载 CLIP 模型", extra={"model": self.model_name, "device": self.device}
        )
        self._model, self._preprocess = clip.load(self.model_name, device=self.device)
        self._model.eval()
        self.logger.info("CLIP 模型加载完成")

    # ── 视觉编码 ──────────────────────────────────────────────────

    def encode_frame(self, image: np.ndarray) -> torch.Tensor:
        """
        单帧编码

        Args:
            image: BGR 图像，shape=(H, W, 3)

        Returns:
            L2 归一化特征向量，shape=(512,)
        """
        return self.encode_batch([image])[0]

    def encode_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        批量编码

        Args:
            images: BGR 图像列表，每张 shape=(H, W, 3)

        Returns:
            L2 归一化特征矩阵，shape=(B, 512)
        """
        self._ensure_loaded()
        from PIL import Image  # noqa: PLC0415

        pil_images = [
            Image.fromarray(img[:, :, ::-1])  # BGR → RGB
            for img in images
        ]
        tensors = torch.stack(
            [self._preprocess(img) for img in pil_images]
        ).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.float()

    # ── 文本编码 ──────────────────────────────────────────────────

    def encode_text(self, text: str) -> torch.Tensor:
        """
        文本编码（用于 VL 融合的语言分支）

        Args:
            text: 场景描述字符串

        Returns:
            L2 归一化特征向量，shape=(512,)
        """
        self._ensure_loaded()
        import clip  # noqa: PLC0415

        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.float()[0]
