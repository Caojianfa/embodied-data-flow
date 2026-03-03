"""HTML 质量报告生成器

输出内容：
  1. 数据概览（帧数、时长、IMU 采样数）
  2. 核心质量指标表（PASS/FAIL 着色）
  3. 时间戳同步误差时序图 + 分布直方图
  4. IMU 加速度 & 角速度时序图
  5. 视频模糊度逐帧曲线
  6. JSON 摘要文件（供程序读取）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，避免 DISPLAY 错误
import matplotlib.pyplot as plt
import numpy as np

from pkg.quality.metrics import QualityReport
from pkg.utils.logger import LoggerMixin


class Reporter(LoggerMixin):
    """
    HTML 质量报告生成器

    Args:
        output_dir: 报告输出目录
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: QualityReport,
        sync_errors: np.ndarray,
        imu_data: np.ndarray,
        video_timestamps_ms: np.ndarray,
        sequence_name: str = "unknown",
    ) -> Path:
        """
        生成完整质量报告

        Args:
            report:              QualityReport 对象
            sync_errors:         每帧同步误差，shape=(N,)，毫秒
            imu_data:            对齐后 IMU 数据，shape=(N, 6)，[wx,wy,wz,ax,ay,az]
            video_timestamps_ms: 视频帧时间戳，shape=(N,)
            sequence_name:       序列名称

        Returns:
            HTML 文件路径
        """
        img_dir = self.output_dir / "img"
        img_dir.mkdir(exist_ok=True)

        chart_paths = {
            "sync_error": self._plot_sync_error(sync_errors, video_timestamps_ms, img_dir),
            "imu_accel": self._plot_imu(imu_data, video_timestamps_ms, img_dir, "accel"),
            "imu_gyro": self._plot_imu(imu_data, video_timestamps_ms, img_dir, "gyro"),
            "blur": self._plot_blur(
                report.blur_variances,
                video_timestamps_ms,
                img_dir,
                report.thresholds.get("min_blur_variance", 100.0),
            ),
        }

        html_path = self.output_dir / f"report_{sequence_name}.html"
        html_path.write_text(
            self._render_html(report, chart_paths, sequence_name), encoding="utf-8"
        )

        # JSON 摘要
        summary = {
            k: v
            for k, v in report.__dict__.items()
            if k not in ("blur_variances", "thresholds")
        }
        summary["passed"] = report.passed()
        summary_path = self.output_dir / f"summary_{sequence_name}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.logger.info(
            "质量报告已生成",
            extra={"html": str(html_path), "json": str(summary_path)},
        )
        return html_path

    # ── 图表生成 ─────────────────────────────────────────────────

    def _plot_sync_error(
        self,
        errors: np.ndarray,
        ts: np.ndarray,
        img_dir: Path,
    ) -> str:
        t_s = (ts - ts[0]) / 1000.0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        ax1.plot(t_s, errors, color="#2196F3", linewidth=0.7, alpha=0.85)
        ax1.axhline(
            np.mean(errors), color="red", linestyle="--", linewidth=1.2,
            label=f"均值 {np.mean(errors):.2f} ms",
        )
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("同步误差 (ms)")
        ax1.set_title("时间戳同步误差时序图")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(errors, bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax2.axvline(
            np.mean(errors), color="red", linestyle="--", linewidth=1.2,
            label=f"均值 {np.mean(errors):.2f} ms",
        )
        ax2.set_xlabel("同步误差 (ms)")
        ax2.set_ylabel("帧数")
        ax2.set_title("同步误差分布直方图")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = img_dir / "sync_error.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    def _plot_imu(
        self,
        imu_data: np.ndarray,
        ts: np.ndarray,
        img_dir: Path,
        mode: str,
    ) -> str:
        t_s = (ts - ts[0]) / 1000.0
        n = min(len(t_s), len(imu_data))

        if mode == "accel":
            data = imu_data[:n, 3:6]
            labels = ["ax", "ay", "az"]
            ylabel = "加速度 (m/s²)"
            title = "加速度时序图"
            fname = "imu_accel.png"
        else:
            data = imu_data[:n, 0:3]
            labels = ["wx", "wy", "wz"]
            ylabel = "角速度 (rad/s)"
            title = "角速度时序图"
            fname = "imu_gyro.png"

        fig, ax = plt.subplots(figsize=(13, 4))
        colors = ["#F44336", "#4CAF50", "#2196F3"]
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(t_s[:n], data[:, i], color=color, linewidth=0.7, alpha=0.85, label=label)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = img_dir / fname
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    def _plot_blur(
        self,
        blur_variances: list[float],
        ts: np.ndarray,
        img_dir: Path,
        threshold: float,
    ) -> str:
        bv = np.array(blur_variances)
        n = min(len(ts), len(bv))
        t_s = (ts[:n] - ts[0]) / 1000.0

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(t_s, bv[:n], color="#9C27B0", linewidth=0.7, alpha=0.85, label="拉普拉斯方差")
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.2,
                   label=f"模糊阈值 {threshold}")
        ax.fill_between(
            t_s, 0, bv[:n], where=(bv[:n] < threshold),
            color="red", alpha=0.25, label="模糊帧",
        )
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("拉普拉斯方差")
        ax.set_title("视频模糊度逐帧检测")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = img_dir / "blur.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(self.output_dir))

    # ── HTML 渲染 ─────────────────────────────────────────────────

    def _render_html(
        self,
        report: QualityReport,
        chart_paths: dict[str, str],
        sequence_name: str,
    ) -> str:
        status_color = "#4CAF50" if report.passed() else "#F44336"
        status_text = "PASS ✓" if report.passed() else "FAIL ✗"
        th = report.thresholds

        def row(label: str, value: str, ok: bool | None = None) -> str:
            if ok is None:
                color = "#333"
            elif ok:
                color = "#43A047"
            else:
                color = "#E53935"
            return (
                f'<tr><td>{label}</td>'
                f'<td style="color:{color};font-weight:600">{value}</td></tr>'
            )

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>质量报告 — {sequence_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; background: #F5F5F5; color: #212121; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .card {{ background: #FFF; border-radius: 10px; padding: 28px 32px;
           margin-bottom: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
  h1 {{ margin: 0 0 8px; font-size: 1.6em; color: #212121; }}
  h2 {{ font-size: 1.1em; color: #424242; border-bottom: 2px solid #EEEEEE;
        padding-bottom: 10px; margin-top: 0; }}
  .badge {{ display: inline-block; padding: 5px 16px; border-radius: 20px;
            color: white; background: {status_color}; font-weight: 700;
            font-size: 0.95em; vertical-align: middle; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 9px 14px; text-align: left; border-bottom: 1px solid #F0F0F0; }}
  th {{ background: #FAFAFA; font-weight: 600; color: #555; font-size: 0.9em; }}
  img {{ max-width: 100%; border-radius: 6px; display: block; margin-top: 8px; }}
  .meta {{ color: #757575; font-size: 0.88em; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">

<div class="card">
  <h1>Embodied Data Flow — 数据质量报告</h1>
  <p class="meta">
    序列：<strong>{sequence_name}</strong> &nbsp;|&nbsp;
    时长：<strong>{report.duration_s:.2f} s</strong> &nbsp;|&nbsp;
    对齐方法：<strong>{report.sync_method}</strong> &nbsp;|&nbsp;
    整体评估：<span class="badge">{status_text}</span>
  </p>
</div>

<div class="card">
  <h2>一、数据概览</h2>
  <table>
    <tr><th>指标</th><th>值</th></tr>
    {row("视频帧数", str(report.total_frames))}
    {row("IMU 采样数", str(report.total_imu_samples))}
    {row("序列时长", f"{report.duration_s:.2f} s")}
    {row("视频帧率", f"{report.video_fps} fps")}
    {row("IMU 采样率", f"{report.imu_rate_hz} Hz")}
    {row("关键帧数", f"{report.keyframe_count}（{report.keyframe_count / max(report.total_frames, 1) * 100:.1f}%）")}
    {row("处理耗时", f"{report.elapsed_s:.2f} s")}
    {row("处理吞吐量", f"{report.throughput_fps} frames/s")}
  </table>
</div>

<div class="card">
  <h2>二、核心质量指标</h2>
  <table>
    <tr><th>指标</th><th>值（阈值）</th></tr>
    {row("帧完整性",
         f"{report.frame_completeness * 100:.2f}%（阈值 ≥ {th.get('min_frame_completeness', 0.99) * 100:.0f}%）",
         report.frame_completeness >= th.get('min_frame_completeness', 0.99))}
    {row("IMU 完整性",
         f"{report.imu_completeness * 100:.2f}%（阈值 ≥ {th.get('min_imu_completeness', 0.995) * 100:.1f}%）",
         report.imu_completeness >= th.get('min_imu_completeness', 0.995))}
    {row("同步误差均值",
         f"{report.sync_error_mean_ms:.3f} ms（阈值 ≤ {th.get('max_sync_error_mean_ms', 5.0)} ms）",
         report.sync_error_mean_ms <= th.get('max_sync_error_mean_ms', 5.0))}
    {row("同步误差标准差",
         f"{report.sync_error_std_ms:.3f} ms（阈值 ≤ {th.get('max_sync_error_std_ms', 3.0)} ms）",
         report.sync_error_std_ms <= th.get('max_sync_error_std_ms', 3.0))}
    {row("同步误差最大值", f"{report.sync_error_max_ms:.3f} ms")}
    {row("视频模糊均值", f"{report.blur_mean:.1f}（拉普拉斯方差，≥ {th.get('min_blur_variance', 100)} 为清晰）")}
    {row("模糊帧占比", f"{report.blurry_frame_rate * 100:.2f}%")}
    {row("IMU 异常值率",
         f"{report.imu_outlier_rate * 100:.4f}%（阈值 ≤ {th.get('max_imu_outlier_rate', 0.001) * 100:.2f}%）",
         report.imu_outlier_rate <= th.get('max_imu_outlier_rate', 0.001))}
  </table>
</div>

<div class="card">
  <h2>三、时间戳对齐分析</h2>
  <img src="{chart_paths['sync_error']}" alt="同步误差图">
</div>

<div class="card">
  <h2>四、IMU 数据时序</h2>
  <img src="{chart_paths['imu_accel']}" alt="加速度时序图">
  <img src="{chart_paths['imu_gyro']}" alt="角速度时序图" style="margin-top:16px">
</div>

<div class="card">
  <h2>五、视频模糊度检测</h2>
  <img src="{chart_paths['blur']}" alt="模糊度逐帧曲线">
</div>

</div>
</body>
</html>"""
