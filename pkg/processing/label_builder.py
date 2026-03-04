"""动作标签构造器

从 Ground Truth 状态序列构造 ActionPredictor 的训练标签。

单步动作定义（7维）：
  action[0:3] — delta_pos:  相邻帧位置差分，单位 m
  action[3:6] — delta_rot:  四元数相对旋转转换为轴角，单位 rad
  action[6]   — gripper:    固定 0.0（无人机无手爪）

标签对齐流程：
  1. 将 GT（200Hz）线性插值到视频帧时间戳（20fps）
  2. 对相邻对齐帧做差分，得到 N-1 个单步动作
  3. 构造 Action Chunking 标签窗口：
       labels[i] = [delta[i], delta[i+1], ..., delta[i+horizon-1]]
       shape = (N-horizon, horizon, 7)
"""
from __future__ import annotations

import numpy as np

from pkg.ingestion.gt_reader import GroundTruth
from pkg.utils.logger import LoggerMixin


class LabelBuilder(LoggerMixin):
    """从 Ground Truth 构造 Action Chunking 训练标签"""

    def __init__(self, action_horizon: int = 10):
        self.action_horizon = action_horizon

    def build(
        self,
        gt: GroundTruth,
        video_timestamps_ms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将 GT 对齐到视频帧时间戳，构造动作标签窗口

        Args:
            gt:                  Ground truth 数据
            video_timestamps_ms: 视频帧时间戳，shape=(N,)，毫秒

        Returns:
            valid_indices: 有效帧索引，shape=(M,)，M = N - horizon
            labels:        动作标签，shape=(M, horizon, 7)
        """
        # 1. GT 插值对齐到视频帧时间戳
        pos_aligned  = self._interpolate(gt.timestamps_ms, gt.positions,   video_timestamps_ms)
        quat_aligned = self._interpolate(gt.timestamps_ms, gt.quaternions,  video_timestamps_ms)

        # 插值后四元数模长略偏离 1，归一化修正
        norms = np.linalg.norm(quat_aligned, axis=1, keepdims=True) + 1e-8
        quat_aligned = quat_aligned / norms

        N       = len(video_timestamps_ms)
        horizon = self.action_horizon

        # 2. 计算相邻帧单步动作，shape=(N-1, 7)
        single_step = self._compute_single_step(pos_aligned, quat_aligned)

        # 3. 构造 Action Chunking 窗口
        #    帧 i 的标签 = single_step[i : i+horizon]
        #    最后 horizon 帧没有足够的未来步骤，剔除
        n_valid = N - horizon
        labels  = np.zeros((n_valid, horizon, 7), dtype=np.float32)
        for i in range(n_valid):
            labels[i] = single_step[i : i + horizon]

        valid_indices = np.arange(n_valid)

        self.logger.info(
            "动作标签构造完成",
            extra={
                "total_frames": N,
                "valid_frames": n_valid,
                "horizon": horizon,
                "pos_delta_mean_mm": round(
                    float(np.abs(labels[:, :, :3]).mean()) * 1000, 4
                ),
                "rot_delta_mean_mrad": round(
                    float(np.abs(labels[:, :, 3:6]).mean()) * 1000, 4
                ),
            },
        )
        return valid_indices, labels

    # ── 内部方法 ──────────────────────────────────────────────────

    def _interpolate(
        self,
        src_ts: np.ndarray,   # (M,)
        src_data: np.ndarray, # (M, D)
        dst_ts: np.ndarray,   # (N,)
    ) -> np.ndarray:
        """
        向量化线性插值，将 src_data 对齐到 dst_ts

        利用 searchsorted 定位区间，单次操作覆盖所有列，O(N log M)
        """
        idx = np.searchsorted(src_ts, dst_ts, side="right")
        idx = np.clip(idx, 1, len(src_ts) - 1)

        t0 = src_ts[idx - 1]
        t1 = src_ts[idx]
        dt = t1 - t0

        # 权重 w：dst_ts 在 [t0, t1] 中的相对位置
        w = np.where(dt > 0, (dst_ts - t0) / dt, 0.0)[:, np.newaxis]  # (N, 1)

        return ((1.0 - w) * src_data[idx - 1] + w * src_data[idx]).astype(np.float64)

    def _compute_single_step(
        self,
        pos: np.ndarray,   # (N, 3)
        quat: np.ndarray,  # (N, 4) [qw, qx, qy, qz]
    ) -> np.ndarray:
        """计算相邻帧的单步动作，shape=(N-1, 7)"""
        delta_pos = (pos[1:] - pos[:-1]).astype(np.float32)               # (N-1, 3)
        delta_rot = self._quat_delta_to_axisangle(quat[:-1], quat[1:])    # (N-1, 3)
        gripper   = np.zeros((len(pos) - 1, 1), dtype=np.float32)         # (N-1, 1)
        return np.concatenate([delta_pos, delta_rot, gripper], axis=1)    # (N-1, 7)

    @staticmethod
    def _quat_delta_to_axisangle(
        q0: np.ndarray,  # (N, 4) [qw, qx, qy, qz]
        q1: np.ndarray,  # (N, 4)
    ) -> np.ndarray:
        """
        计算从 q0 到 q1 的相对旋转，转换为轴角表示

        步骤：
          1. q_delta = q0^{-1} ⊗ q1
             （单位四元数的逆 = 共轭：[qw, -qx, -qy, -qz]）
          2. 确保 qw >= 0，取旋转最短路径
          3. angle = 2 * arccos(qw)
          4. axis  = [qx, qy, qz] / sin(angle/2)
          5. 轴角  = axis * angle，shape=(N, 3)
        """
        # q0 的共轭（即逆）
        q0_inv = q0 * np.array([1.0, -1.0, -1.0, -1.0])

        # 四元数乘法：q_delta = q0_inv ⊗ q1
        aw, ax, ay, az = q0_inv[:, 0], q0_inv[:, 1], q0_inv[:, 2], q0_inv[:, 3]
        bw, bx, by, bz = q1[:, 0],    q1[:, 1],    q1[:, 2],    q1[:, 3]

        dw = aw * bw - ax * bx - ay * by - az * bz
        dx = aw * bx + ax * bw + ay * bz - az * by
        dy = aw * by - ax * bz + ay * bw + az * bx
        dz = aw * bz + ax * by - ay * bx + az * bw

        # 确保 qw >= 0，使旋转角落在 [0, π] 取最短弧
        sign = np.where(dw >= 0, 1.0, -1.0)
        dw, dx, dy, dz = dw * sign, dx * sign, dy * sign, dz * sign

        # 四元数 → 轴角
        dw_clamped = np.clip(dw, -1.0, 1.0)
        angle      = 2.0 * np.arccos(dw_clamped)               # (N,)
        sin_half   = np.sqrt(np.maximum(1.0 - dw_clamped ** 2, 0.0))

        # 旋转角极小时（<1e-7 rad）轴方向无意义，置零
        eps   = 1e-7
        valid = sin_half > eps
        inv_s = np.where(valid, 1.0 / (sin_half + 1e-10), 0.0)

        axis_x = dx * inv_s
        axis_y = dy * inv_s
        axis_z = dz * inv_s

        axisangle = np.stack(
            [axis_x * angle, axis_y * angle, axis_z * angle], axis=1
        )
        return axisangle.astype(np.float32)
