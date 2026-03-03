"""帧预处理模块

功能：
  1. Resize 到目标分辨率（默认 224×224，CLIP 输入尺寸）
  2. ImageNet 归一化，转 CHW float32，用于模型输入
  3. 模糊度检测（拉普拉斯方差）
  4. 关键帧提取（帧间均方差阈值）
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pkg.ingestion.video_reader import VideoFrame
from pkg.utils.logger import LoggerMixin


@dataclass
class ProcessedFrame:
    """预处理后的帧"""

    timestamp_ms: float
    frame_id: int
    image_resized: np.ndarray       # BGR，shape=(H, W, 3)，uint8，用于可视化
    image_normalized: np.ndarray    # RGB，shape=(3, H, W)，float32，用于模型输入
    blur_variance: float            # 拉普拉斯方差，越小越模糊
    is_keyframe: bool


class FrameProcessor(LoggerMixin):
    """
    帧预处理器

    Args:
        target_height:           目标高度（默认 224）
        target_width:            目标宽度（默认 224）
        blur_threshold:          低于此拉普拉斯方差视为模糊帧（默认 100.0）
        keyframe_diff_threshold: 帧间归一化均方差超过此值视为关键帧（默认 0.02）
    """

    # ImageNet 归一化参数（RGB 通道顺序）
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        target_height: int = 224,
        target_width: int = 224,
        blur_threshold: float = 100.0,
        keyframe_diff_threshold: float = 0.02,
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.blur_threshold = blur_threshold
        self.keyframe_diff_threshold = keyframe_diff_threshold
        self._prev_gray: np.ndarray | None = None

    def process(self, frame: VideoFrame) -> ProcessedFrame:
        """处理单帧"""
        # 1. Resize
        resized = cv2.resize(
            frame.image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. 模糊度检测（基于灰度图拉普拉斯方差）
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. 关键帧检测
        is_keyframe = self._detect_keyframe(gray)

        # 4. 归一化：BGR → RGB → float32 → ImageNet normalize → CHW
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD          # HWC，float32
        normalized = normalized.transpose(2, 0, 1)         # → CHW

        return ProcessedFrame(
            timestamp_ms=frame.timestamp_ms,
            frame_id=frame.frame_id,
            image_resized=resized,
            image_normalized=normalized.astype(np.float32),
            blur_variance=blur_variance,
            is_keyframe=is_keyframe,
        )

    def _detect_keyframe(self, gray: np.ndarray) -> bool:
        """
        基于帧间归一化均方差的关键帧检测。
        第一帧始终为关键帧；后续帧与上一帧差异超过阈值时为关键帧。
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        diff = np.mean(
            np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32))
        ) / 255.0
        self._prev_gray = gray
        return diff > self.keyframe_diff_threshold

    def reset(self) -> None:
        """重置状态，处理新序列前调用"""
        self._prev_gray = None
