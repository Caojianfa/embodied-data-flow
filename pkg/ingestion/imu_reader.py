"""EuRoC MAV IMU 数据读取器

EuRoC imu0/data.csv 格式（纳秒时间戳）：
  #timestamp [ns], w_x [rad/s], w_y, w_z, a_x [m/s²], a_y, a_z
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pkg.utils.logger import LoggerMixin


@dataclass
class ImuData:
    """IMU 数据集合"""

    timestamps_ms: np.ndarray  # shape=(N,)，毫秒
    gyro: np.ndarray           # shape=(N, 3)，角速度 rad/s，[wx, wy, wz]
    accel: np.ndarray          # shape=(N, 3)，加速度 m/s²，[ax, ay, az]
    rate_hz: float

    @property
    def total_samples(self) -> int:
        return len(self.timestamps_ms)

    @property
    def duration_s(self) -> float:
        if self.total_samples < 2:
            return 0.0
        return (self.timestamps_ms[-1] - self.timestamps_ms[0]) / 1000.0

    def as_array(self) -> np.ndarray:
        """返回合并数组，shape=(N, 6)，列顺序 [wx,wy,wz,ax,ay,az]"""
        return np.concatenate([self.gyro, self.accel], axis=1)


class ImuReader(LoggerMixin):
    """
    EuRoC MAV IMU 数据读取器

    读取 mav0/imu0/data.csv，将时间戳从纳秒转换为毫秒。

    Args:
        sequence_dir: EuRoC 序列根目录（含 mav0/ 子目录）
        rate_hz:      IMU 采样率（EuRoC = 200Hz）
    """

    def __init__(self, sequence_dir: str | Path, rate_hz: float = 200.0):
        self.sequence_dir = Path(sequence_dir)
        self.rate_hz = rate_hz
        self._imu_csv = self.sequence_dir / "mav0" / "imu0" / "data.csv"

    def load(self) -> ImuData:
        """
        加载全部 IMU 数据

        Returns:
            ImuData 对象
        """
        if not self._imu_csv.exists():
            raise FileNotFoundError(f"IMU 数据文件不存在: {self._imu_csv}")

        # EuRoC IMU CSV 第一行为注释，用 comment='#' 跳过
        df = pd.read_csv(
            self._imu_csv,
            comment="#",
            header=None,
            names=["timestamp_ns", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"],
            dtype=float,
        )

        timestamps_ms = df["timestamp_ns"].values / 1e6
        gyro = df[["w_x", "w_y", "w_z"]].values.astype(np.float64)
        accel = df[["a_x", "a_y", "a_z"]].values.astype(np.float64)

        imu = ImuData(
            timestamps_ms=timestamps_ms,
            gyro=gyro,
            accel=accel,
            rate_hz=self.rate_hz,
        )

        self.logger.info(
            "IMU 数据加载完成",
            extra={
                "samples": imu.total_samples,
                "rate_hz": imu.rate_hz,
                "duration_s": round(imu.duration_s, 2),
            },
        )
        return imu
