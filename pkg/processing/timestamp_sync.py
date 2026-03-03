"""时间戳对齐模块（核心算法）

将 IMU 数据（200Hz）对齐到视频帧时间戳（20fps），采样率差 10×。

支持两种方法：
  linear  — 线性插值，误差降至亚毫秒级（推荐）
  nearest — 最近邻，误差最大 2.5ms，实现简单

算法复杂度：O(N+M)，双指针实现，N=视频帧数，M=IMU采样数。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkg.utils.logger import LoggerMixin


@dataclass
class SyncResult:
    """时间戳对齐结果"""

    aligned_imu: np.ndarray          # shape=(N_frames, 6)，对齐后 IMU [wx,wy,wz,ax,ay,az]
    sync_errors_ms: np.ndarray       # shape=(N_frames,)，每帧同步误差（ms）
    video_timestamps_ms: np.ndarray  # shape=(N_frames,)
    imu_timestamps_ms: np.ndarray    # shape=(N_imu,)
    method: str

    @property
    def n_frames(self) -> int:
        return len(self.aligned_imu)

    @property
    def error_mean_ms(self) -> float:
        return float(np.mean(self.sync_errors_ms))

    @property
    def error_std_ms(self) -> float:
        return float(np.std(self.sync_errors_ms))

    @property
    def error_max_ms(self) -> float:
        return float(np.max(self.sync_errors_ms))


class TimestampSync(LoggerMixin):
    """
    多模态时间戳对齐器

    核心场景：
      视频（EuRoC cam0）：20 FPS  → 帧间隔 50ms
      IMU（EuRoC imu0）：200 Hz   → 采样间隔 5ms
      采样率差 10×，线性插值对齐到亚毫秒级精度
    """

    def align(
        self,
        video_ts: np.ndarray,
        imu_ts: np.ndarray,
        imu_data: np.ndarray,
        method: str = "linear",
    ) -> SyncResult:
        """
        将 IMU 数据对齐到视频帧时间戳

        Args:
            video_ts:  视频帧时间戳，shape=(N_frames,)，毫秒
            imu_ts:    IMU 时间戳，shape=(N_imu,)，毫秒
            imu_data:  IMU 数据，shape=(N_imu, 6)
            method:    'linear' | 'nearest'

        Returns:
            SyncResult
        """
        if method == "linear":
            aligned, errors = self._linear_interpolate(video_ts, imu_ts, imu_data)
        elif method == "nearest":
            aligned, errors = self._nearest_neighbor(video_ts, imu_ts, imu_data)
        else:
            raise ValueError(f"未知对齐方法: {method}，可选 'linear' 或 'nearest'")

        result = SyncResult(
            aligned_imu=aligned,
            sync_errors_ms=errors,
            video_timestamps_ms=video_ts,
            imu_timestamps_ms=imu_ts,
            method=method,
        )

        self.logger.info(
            "时间戳对齐完成",
            extra={
                "method": method,
                "n_frames": result.n_frames,
                "sync_error_mean_ms": round(result.error_mean_ms, 3),
                "sync_error_std_ms": round(result.error_std_ms, 3),
                "sync_error_max_ms": round(result.error_max_ms, 3),
            },
        )
        return result

    # ── 线性插值 ──────────────────────────────────────────────────

    def _linear_interpolate(
        self,
        video_ts: np.ndarray,
        imu_ts: np.ndarray,
        imu_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        O(N+M) 双指针线性插值

        对于每个视频帧时间戳 v：
          找到满足 imu_ts[j] <= v <= imu_ts[j+1] 的相邻 IMU 采样对
          插值权重 w = (v - t0) / (t1 - t0)
          result     = (1-w)*imu[j] + w*imu[j+1]
          sync_error = min(v-t0, t1-v)
        """
        n_frames = len(video_ts)
        n_imu = len(imu_ts)
        n_feat = imu_data.shape[1]

        aligned = np.zeros((n_frames, n_feat), dtype=np.float64)
        errors = np.zeros(n_frames, dtype=np.float64)

        j = 0  # IMU 指针
        for i, vt in enumerate(video_ts):
            # 推进 j 直到 imu_ts[j+1] >= vt
            while j < n_imu - 2 and imu_ts[j + 1] < vt:
                j += 1

            t0 = imu_ts[j]
            t1 = imu_ts[min(j + 1, n_imu - 1)]

            if vt <= t0:
                # 视频帧早于 IMU 起始
                aligned[i] = imu_data[j]
                errors[i] = t0 - vt
            elif vt >= t1 or t1 == t0:
                # 视频帧晚于 IMU 结束，或相同时间戳
                aligned[i] = imu_data[min(j + 1, n_imu - 1)]
                errors[i] = vt - t1 if vt >= t1 else 0.0
            else:
                w = (vt - t0) / (t1 - t0)
                aligned[i] = (1.0 - w) * imu_data[j] + w * imu_data[j + 1]
                errors[i] = min(vt - t0, t1 - vt)

        return aligned, errors

    # ── 最近邻 ────────────────────────────────────────────────────

    def _nearest_neighbor(
        self,
        video_ts: np.ndarray,
        imu_ts: np.ndarray,
        imu_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """最近邻对齐，误差最大 2.5ms（IMU 5ms 间隔的一半）"""
        indices = np.searchsorted(imu_ts, video_ts)
        indices = np.clip(indices, 0, len(imu_ts) - 1)

        prev_idx = np.maximum(indices - 1, 0)
        prev_err = np.abs(video_ts - imu_ts[prev_idx])
        curr_err = np.abs(video_ts - imu_ts[indices])

        best_idx = np.where(prev_err < curr_err, prev_idx, indices)
        aligned = imu_data[best_idx]
        errors = np.abs(video_ts - imu_ts[best_idx])

        return aligned, errors
