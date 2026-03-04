"""EuRoC Ground Truth 状态读取器

读取 state_groundtruth_estimate0/data.csv，包含 17 列：
  timestamp (ns)
  p_RS_R_x/y/z [m]           — 全局坐标系位置
  q_RS_w/x/y/z []            — 单位四元数姿态
  v_RS_R_x/y/z [m/s]         — 全局坐标系速度
  b_w_RS_S_x/y/z [rad/s]     — 陀螺仪偏置
  b_a_RS_S_x/y/z [m/s²]      — 加速度计偏置
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pkg.utils.logger import LoggerMixin


@dataclass
class GroundTruth:
    """Ground Truth 状态数据"""

    timestamps_ms: np.ndarray  # shape=(N,)，毫秒
    positions: np.ndarray      # shape=(N, 3)，[px, py, pz]，米
    quaternions: np.ndarray    # shape=(N, 4)，[qw, qx, qy, qz]，单位四元数
    velocities: np.ndarray     # shape=(N, 3)，[vx, vy, vz]，m/s


class GtReader(LoggerMixin):
    """EuRoC Ground Truth 读取器"""

    GT_SUBPATH = "mav0/state_groundtruth_estimate0/data.csv"

    def __init__(self, sequence_dir: Path):
        self.gt_csv = Path(sequence_dir) / self.GT_SUBPATH

    def load(self) -> GroundTruth:
        """加载 Ground Truth CSV，返回 GroundTruth 数据类"""
        if not self.gt_csv.exists():
            raise FileNotFoundError(
                f"Ground truth 文件不存在: {self.gt_csv}\n"
                "请确认数据集已下载：bash scripts/download_data.sh"
            )

        data = np.genfromtxt(self.gt_csv, delimiter=",", skip_header=1)

        gt = GroundTruth(
            timestamps_ms=data[:, 0] / 1e6,  # 纳秒 → 毫秒
            positions=data[:, 1:4],           # px, py, pz
            quaternions=data[:, 4:8],         # qw, qx, qy, qz
            velocities=data[:, 8:11],         # vx, vy, vz
        )

        self.logger.info(
            "Ground truth 加载完成",
            extra={
                "rows": len(gt.timestamps_ms),
                "duration_s": round(
                    (gt.timestamps_ms[-1] - gt.timestamps_ms[0]) / 1000.0, 2
                ),
                "pos_range_x": (round(float(gt.positions[:, 0].min()), 3),
                                round(float(gt.positions[:, 0].max()), 3)),
            },
        )
        return gt
