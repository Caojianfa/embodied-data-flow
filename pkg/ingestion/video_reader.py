"""EuRoC MAV 图像序列读取器

EuRoC cam0 格式：
  mav0/cam0/data.csv   — 每行：timestamp_ns,filename.png
  mav0/cam0/data/      — PNG 图像文件目录

设计：流式 yield 单帧，避免一次性加载全部图像导致 OOM，
借鉴 Flink/Kafka 生产者-消费者模式。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from pkg.utils.logger import LoggerMixin


@dataclass
class VideoFrame:
    """单帧数据结构"""

    timestamp_ms: float    # 时间戳（毫秒，从 EuRoC 纳秒转换）
    frame_id: int          # 帧序号（从 0 开始）
    image: np.ndarray      # BGR 图像，shape=(H, W, 3)
    fps: float             # 原始帧率（EuRoC cam0 = 20fps）


class VideoReader(LoggerMixin):
    """
    EuRoC MAV 图像序列流式读取器

    Args:
        sequence_dir: EuRoC 序列根目录（含 mav0/ 子目录）
        fps:          相机帧率，EuRoC cam0 = 20fps
    """

    def __init__(self, sequence_dir: str | Path, fps: float = 20.0):
        self.sequence_dir = Path(sequence_dir)
        self.fps = fps
        self._cam0_csv = self.sequence_dir / "mav0" / "cam0" / "data.csv"
        self._cam0_dir = self.sequence_dir / "mav0" / "cam0" / "data"
        self._timestamps: list[float] = []
        self._image_paths: list[Path] = []
        self._loaded = False

    # ── 索引加载（惰性） ──────────────────────────────────────────

    def _load_index(self) -> None:
        """读取 cam0/data.csv，建立时间戳 → 图像路径的映射"""
        if self._loaded:
            return
        if not self._cam0_csv.exists():
            raise FileNotFoundError(f"cam0 时间戳文件不存在: {self._cam0_csv}")

        with open(self._cam0_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                ts_ms = ts_ns / 1e6
                # EuRoC data.csv 第二列为文件名（如 1403636579758555392.png）
                filename = row[1].strip() if len(row) > 1 else f"{ts_ns}.png"
                self._timestamps.append(ts_ms)
                self._image_paths.append(self._cam0_dir / filename)

        self._loaded = True
        self.logger.info(
            "cam0 索引加载完成",
            extra={"frames": len(self._timestamps), "fps": self.fps},
        )

    # ── 属性 ─────────────────────────────────────────────────────

    @property
    def total_frames(self) -> int:
        self._load_index()
        return len(self._timestamps)

    @property
    def timestamps_ms(self) -> list[float]:
        self._load_index()
        return self._timestamps

    @property
    def duration_s(self) -> float:
        self._load_index()
        if len(self._timestamps) < 2:
            return 0.0
        return (self._timestamps[-1] - self._timestamps[0]) / 1000.0

    # ── 流式读取 ──────────────────────────────────────────────────

    def stream_frames(self, max_frames: int | None = None) -> Iterator[VideoFrame]:
        """
        流式读取图像帧，每次 yield 一帧。

        Args:
            max_frames: 最多读取帧数，None 表示全部

        Yields:
            VideoFrame
        """
        self._load_index()
        limit = max_frames if max_frames is not None else len(self._timestamps)

        for frame_id, (ts_ms, img_path) in enumerate(
            zip(self._timestamps[:limit], self._image_paths[:limit])
        ):
            if not img_path.exists():
                self.logger.warning(
                    "图像文件缺失，跳过",
                    extra={"frame_id": frame_id, "path": str(img_path)},
                )
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.warning(
                    "图像读取失败，跳过",
                    extra={"frame_id": frame_id, "path": str(img_path)},
                )
                continue

            yield VideoFrame(
                timestamp_ms=ts_ms,
                frame_id=frame_id,
                image=image,
                fps=self.fps,
            )
