"""数据质量评估模块

计算以下维度的质量指标：
  - 帧/IMU 完整性（actual vs expected）
  - 时间戳对齐误差分布
  - 视频模糊度（拉普拉斯方差）
  - IMU 异常值率（滑动窗口 Nσ 准则，σ 倍数可配置）
  - 处理吞吐量（frames/s）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pkg.processing.timestamp_sync import SyncResult
from pkg.utils.logger import LoggerMixin


@dataclass
class QualityReport:
    """质量评估报告（所有指标汇总）"""

    # 数据概览
    total_frames: int = 0
    total_imu_samples: int = 0
    duration_s: float = 0.0
    video_fps: float = 0.0
    imu_rate_hz: float = 0.0

    # 完整性
    frame_completeness: float = 0.0   # actual / expected，理想值 > 0.99
    frame_drop_rate: float = 0.0
    imu_completeness: float = 0.0
    imu_outlier_rate: float = 0.0     # 超过 3σ 的采样点比例

    # 时间戳对齐
    sync_error_mean_ms: float = 0.0
    sync_error_std_ms: float = 0.0
    sync_error_max_ms: float = 0.0
    sync_method: str = ""

    # 视频质量
    blur_variances: list[float] = field(default_factory=list)
    blur_mean: float = 0.0
    blur_min: float = 0.0
    blurry_frame_rate: float = 0.0    # 模糊帧（< threshold）占比
    keyframe_count: int = 0

    # 性能
    throughput_fps: float = 0.0
    elapsed_s: float = 0.0

    # 阈值（用于 pass/fail 判断）
    thresholds: dict = field(default_factory=dict)

    def passed(self) -> bool:
        """全部核心指标是否通过"""
        th = self.thresholds
        checks = [
            self.frame_completeness >= th.get("min_frame_completeness", 0.99),
            self.imu_completeness >= th.get("min_imu_completeness", 0.995),
            self.sync_error_mean_ms <= th.get("max_sync_error_mean_ms", 5.0),
            self.sync_error_std_ms <= th.get("max_sync_error_std_ms", 3.0),
            self.imu_outlier_rate <= th.get("max_imu_outlier_rate", 0.001),
        ]
        return all(checks)


class QualityMetrics(LoggerMixin):
    """数据质量评估器"""

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = thresholds or {
            "min_frame_completeness": 0.99,
            "min_imu_completeness": 0.995,
            "max_sync_error_mean_ms": 5.0,
            "max_sync_error_std_ms": 3.0,
            "min_blur_variance": 100.0,
            "max_imu_outlier_rate": 0.001,
            "target_throughput_fps": 100.0,
        }

    def evaluate(
        self,
        sync_result: SyncResult,
        blur_variances: list[float],
        keyframe_count: int,
        imu_data_raw: np.ndarray,
        video_fps: float,
        imu_rate_hz: float,
        elapsed_s: float,
    ) -> QualityReport:
        """
        计算全部质量指标

        Args:
            sync_result:     时间戳对齐结果
            blur_variances:  每帧拉普拉斯方差列表
            keyframe_count:  关键帧数量
            imu_data_raw:    原始 IMU 数据，shape=(N, 6)
            video_fps:       视频帧率
            imu_rate_hz:     IMU 采样率
            elapsed_s:       Pipeline 总耗时（秒）

        Returns:
            QualityReport
        """
        n_frames = sync_result.n_frames
        n_imu = len(sync_result.imu_timestamps_ms)

        duration_s = (
            (sync_result.video_timestamps_ms[-1] - sync_result.video_timestamps_ms[0])
            / 1000.0
            if n_frames > 1
            else 0.0
        )

        # 完整性
        expected_frames = max(1, int(round(duration_s * video_fps)))
        frame_completeness = min(1.0, n_frames / expected_frames)

        expected_imu = max(1, int(round(duration_s * imu_rate_hz)))
        imu_completeness = min(1.0, n_imu / expected_imu)

        # IMU 异常值率（滑动窗口 Nσ 准则，逐轴检测）
        # 用局部统计量替代全局均值，避免把正常飞行机动误判为异常
        n_imu_raw = len(imu_data_raw)
        window_s = self.thresholds.get("imu_outlier_window_s", 2.0)
        sigma = self.thresholds.get("imu_outlier_sigma", 4.5)
        w = max(2, int(round(imu_rate_hz * window_s)))
        half_w = w // 2

        # 边界 edge 填充，首尾窗口使用端点值补齐
        padded = np.pad(imu_data_raw, ((half_w, half_w), (0, 0)), mode="edge")

        # O(N) cumsum 滑动窗口：E[X] 和 E[X²]
        cs = np.zeros((len(padded) + 1, imu_data_raw.shape[1]))
        cs2 = np.zeros_like(cs)
        cs[1:] = np.cumsum(padded, axis=0)
        cs2[1:] = np.cumsum(padded ** 2, axis=0)

        local_mean = (cs[w:w + n_imu_raw] - cs[:n_imu_raw]) / w
        local_var = (cs2[w:w + n_imu_raw] - cs2[:n_imu_raw]) / w - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0.0)) + 1e-8

        outlier_mask = np.any(
            np.abs(imu_data_raw - local_mean) > sigma * local_std, axis=1
        )
        imu_outlier_rate = float(np.mean(outlier_mask))

        # 视频模糊度
        bv = np.array(blur_variances) if blur_variances else np.array([0.0])
        blur_threshold = self.thresholds.get("min_blur_variance", 100.0)

        report = QualityReport(
            total_frames=n_frames,
            total_imu_samples=n_imu,
            duration_s=round(duration_s, 2),
            video_fps=video_fps,
            imu_rate_hz=imu_rate_hz,
            frame_completeness=round(frame_completeness, 4),
            frame_drop_rate=round(1.0 - frame_completeness, 4),
            imu_completeness=round(imu_completeness, 4),
            imu_outlier_rate=round(imu_outlier_rate, 6),
            sync_error_mean_ms=round(sync_result.error_mean_ms, 3),
            sync_error_std_ms=round(sync_result.error_std_ms, 3),
            sync_error_max_ms=round(sync_result.error_max_ms, 3),
            sync_method=sync_result.method,
            blur_variances=blur_variances,
            blur_mean=round(float(np.mean(bv)), 2),
            blur_min=round(float(np.min(bv)), 2),
            blurry_frame_rate=round(float(np.mean(bv < blur_threshold)), 4),
            keyframe_count=keyframe_count,
            throughput_fps=round(n_frames / elapsed_s, 1) if elapsed_s > 0 else 0.0,
            elapsed_s=round(elapsed_s, 2),
            thresholds=self.thresholds,
        )

        status = "PASS" if report.passed() else "FAIL"
        self.logger.info(
            f"质量评估完成 [{status}]",
            extra={
                "frame_completeness": report.frame_completeness,
                "imu_completeness": report.imu_completeness,
                "sync_error_mean_ms": report.sync_error_mean_ms,
                "sync_error_std_ms": report.sync_error_std_ms,
                "blur_mean": report.blur_mean,
                "imu_outlier_rate": report.imu_outlier_rate,
                "throughput_fps": report.throughput_fps,
                "passed": report.passed(),
            },
        )
        return report
