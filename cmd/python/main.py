"""Embodied Data Flow — Pipeline 入口

用法：
  # 使用默认配置（处理 MH_01_easy 全部帧）
  python cmd/python/main.py

  # 指定序列 + 限制帧数（调试用）
  python cmd/python/main.py --sequence MH_02_easy --max-frames 200

  # 指定对齐方法
  python cmd/python/main.py --sync-method nearest

  # 不保存中间 npy 文件
  python cmd/python/main.py --no-save

Pipeline 阶段：
  1. 数据摄取    — VideoReader 流式读取 PNG 序列，ImuReader 加载 CSV
  2. 帧处理      — resize、归一化、模糊度检测、关键帧提取
  3. 时间戳对齐  — 线性插值，将 IMU(200Hz) 对齐到视频帧(20fps)
  4. 视觉编码    — CLIP ViT-B/32 批量编码，输出 512 维特征
  5. VL 融合     — Cross-Attention 融合视觉与场景描述语言特征
  6. 动作预测    — Transformer Encoder，输出 (N, T, 7) 动作序列
  7. 质量评估    — 计算完整性、同步误差、模糊度等指标
  8. 报告生成    — HTML 报告 + JSON 摘要 + npy 中间结果
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 将项目根目录加入 sys.path，支持直接执行
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.ingestion.imu_reader import ImuReader
from pkg.ingestion.video_reader import VideoReader
from pkg.models.action_predictor import ActionPredictor
from pkg.models.vision_encoder import VisionEncoder
from pkg.models.vl_fusion import VisionLanguageFusion
from pkg.processing.frame_processor import FrameProcessor
from pkg.processing.timestamp_sync import TimestampSync
from pkg.quality.metrics import QualityMetrics
from pkg.quality.reporter import Reporter
from pkg.utils.config import load_config
from pkg.utils.logger import get_logger

logger = get_logger(__name__)


# ── CLI 参数 ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embodied Data Flow — 具身智能多模态数据处理 Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="配置文件路径（默认 configs/pipeline.yaml）",
    )
    parser.add_argument(
        "--sequence", default=None,
        help="覆盖配置中的序列名，如 MH_02_easy",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="最多处理帧数，默认处理全部（调试时建议设 200）",
    )
    parser.add_argument(
        "--sync-method", choices=["linear", "nearest"], default="linear",
        help="时间戳对齐方法（默认 linear，更精确；nearest 更快）",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="不保存对齐后的 npy 中间文件",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="加载训练好的模型权重，如 data/output/checkpoints/MH_01_easy_best.pt",
    )
    return parser.parse_args()


# ── Pipeline 主函数 ──────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    # 加载配置
    cfg = load_config(args.config)
    sequence = args.sequence or cfg.data.sequence
    max_frames = args.max_frames if args.max_frames is not None else cfg.pipeline.max_frames
    sequence_dir = Path(cfg.data.euroc_base) / sequence

    if not sequence_dir.exists():
        logger.error(
            "序列目录不存在，请先下载数据",
            extra={"path": str(sequence_dir), "hint": "bash scripts/download_data.sh"},
        )
        sys.exit(1)

    logger.info(
        "Pipeline 启动",
        extra={
            "config": args.config,
            "sequence": sequence,
            "max_frames": max_frames,
            "sync_method": args.sync_method,
            "device": cfg.model.device,
        },
    )
    pipeline_start = time.time()

    # ── 1. 数据摄取 ───────────────────────────────────────────────

    video_reader = VideoReader(sequence_dir, fps=float(cfg.video.fps))
    imu_reader = ImuReader(sequence_dir, rate_hz=float(cfg.imu.rate_hz))

    imu_data = imu_reader.load()

    logger.info(
        "开始流式读取视频帧",
        extra={
            "total_frames": video_reader.total_frames,
            "fps": video_reader.fps,
            "duration_s": round(video_reader.duration_s, 2),
        },
    )

    # ── 2. 帧处理（流式） ─────────────────────────────────────────

    frame_processor = FrameProcessor(
        target_height=int(cfg.video.target_height),
        target_width=int(cfg.video.target_width),
        blur_threshold=float(cfg.frame.blur_threshold),
        keyframe_diff_threshold=float(cfg.frame.keyframe_diff_threshold),
    )

    processed_frames = []
    blur_variances: list[float] = []
    keyframe_count = 0

    for frame in video_reader.stream_frames(max_frames=max_frames):
        pf = frame_processor.process(frame)
        processed_frames.append(pf)
        blur_variances.append(pf.blur_variance)
        if pf.is_keyframe:
            keyframe_count += 1

    n_frames = len(processed_frames)
    if n_frames == 0:
        logger.error("未读取到任何帧，请检查数据目录结构")
        sys.exit(1)

    logger.info(
        "帧处理完成",
        extra={
            "processed_frames": n_frames,
            "keyframes": keyframe_count,
            "blurry_frames": sum(1 for v in blur_variances if v < float(cfg.frame.blur_threshold)),
        },
    )

    # ── 3. 时间戳对齐 ─────────────────────────────────────────────

    video_ts = np.array([pf.timestamp_ms for pf in processed_frames])
    syncer = TimestampSync()
    sync_result = syncer.align(
        video_ts=video_ts,
        imu_ts=imu_data.timestamps_ms,
        imu_data=imu_data.as_array(),
        method=args.sync_method,
    )

    # ── 4. 视觉编码 ───────────────────────────────────────────────

    vision_encoder = VisionEncoder(
        model_name=str(cfg.model.clip_model),
        device=str(cfg.model.device),
    )

    batch_size = int(cfg.pipeline.batch_size)
    all_visual_feats: list[torch.Tensor] = []

    logger.info("开始视觉编码", extra={"batch_size": batch_size, "total_frames": n_frames})
    vision_start = time.time()

    for i in range(0, n_frames, batch_size):
        batch = [pf.image_resized for pf in processed_frames[i: i + batch_size]]
        feats = vision_encoder.encode_batch(batch)
        all_visual_feats.append(feats)
        if (i // batch_size) % 10 == 0:
            logger.info(f"视觉编码进度 {min(i + batch_size, n_frames)}/{n_frames}")

    visual_feats = torch.cat(all_visual_feats, dim=0)  # (N, 512)
    vision_elapsed = time.time() - vision_start
    logger.info(
        "视觉编码完成",
        extra={
            "throughput_fps": round(n_frames / vision_elapsed, 1),
            "feat_shape": list(visual_feats.shape),
        },
    )

    # ── 5. VL 融合 ────────────────────────────────────────────────

    # 从配置获取当前序列的场景描述（用于语言分支输入）
    lang_map: dict = cfg.language.to_dict()
    language_text: str = lang_map.get(sequence, str(cfg.language.default))

    text_feat = vision_encoder.encode_text(language_text)  # (512,)

    vl_fusion = VisionLanguageFusion(dim=int(cfg.model.vision_dim))
    predictor = ActionPredictor(
        fusion_dim=int(cfg.model.vision_dim),
        imu_window=int(cfg.model.imu_window),
        action_dim=int(cfg.model.action_dim),
        action_horizon=int(cfg.model.action_horizon),
        d_model=int(cfg.model.transformer_dim),
        nhead=int(cfg.model.transformer_heads),
        num_layers=int(cfg.model.transformer_layers),
    )

    # 加载训练权重（可选）
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=cfg.model.device)
        vl_fusion.load_state_dict(ckpt["vl_fusion"])
        predictor.load_state_dict(ckpt["action_predictor"])
        logger.info(
            "已加载训练权重",
            extra={
                "checkpoint": args.checkpoint,
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
            },
        )
    else:
        logger.warning("未指定 --checkpoint，使用随机初始化权重，动作预测结果无意义")

    vl_fusion.eval()
    predictor.eval()

    with torch.no_grad():
        fused_feats = vl_fusion(visual_feats, text_feat)  # (N, 512)

    logger.info(
        "VL 融合完成",
        extra={"language": language_text, "fused_shape": list(fused_feats.shape)},
    )

    # ── 6. 动作预测 ───────────────────────────────────────────────

    # 为每帧构建 IMU 历史窗口（滑动窗口）
    imu_window_size = int(cfg.model.imu_window)
    aligned_imu = torch.from_numpy(sync_result.aligned_imu).float()  # (N, 6)

    imu_windows: list[torch.Tensor] = []
    for i in range(n_frames):
        start = max(0, i - imu_window_size + 1)
        window = aligned_imu[start: i + 1]  # (≤W, 6)
        if len(window) < imu_window_size:
            # 首帧不足时，用第 0 帧填充
            pad = aligned_imu[0:1].expand(imu_window_size - len(window), -1)
            window = torch.cat([pad, window], dim=0)
        imu_windows.append(window)

    imu_windows_tensor = torch.stack(imu_windows, dim=0)  # (N, W, 6)

    with torch.no_grad():
        actions = predictor.predict(fused_feats, imu_windows_tensor)  # (N, T, 7)

    logger.info("动作预测完成", extra={"actions_shape": list(actions.shape)})

    # ── 7. 质量评估 & 报告 ────────────────────────────────────────

    pipeline_elapsed = time.time() - pipeline_start

    quality = QualityMetrics(thresholds=cfg.quality.to_dict())
    quality_report = quality.evaluate(
        sync_result=sync_result,
        blur_variances=blur_variances,
        keyframe_count=keyframe_count,
        imu_data_raw=imu_data.as_array(),
        video_fps=float(cfg.video.fps),
        imu_rate_hz=float(cfg.imu.rate_hz),
        elapsed_s=pipeline_elapsed,
    )

    reporter = Reporter(output_dir=cfg.data.reports_dir)
    html_path = reporter.generate(
        report=quality_report,
        sync_errors=sync_result.sync_errors_ms,
        imu_data=sync_result.aligned_imu,
        video_timestamps_ms=video_ts,
        sequence_name=sequence,
    )

    # ── 8. 保存中间结果 ───────────────────────────────────────────

    save = (not args.no_save) and bool(cfg.pipeline.save_aligned)
    if save:
        aligned_dir = Path(cfg.data.aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        prefix = aligned_dir / sequence
        np.save(f"{prefix}_aligned_imu.npy", sync_result.aligned_imu)
        np.save(f"{prefix}_visual_feats.npy", visual_feats.numpy())
        np.save(f"{prefix}_fused_feats.npy", fused_feats.numpy())
        np.save(f"{prefix}_actions.npy", actions.numpy())
        np.save(f"{prefix}_video_ts.npy", video_ts)

        logger.info("中间结果已保存", extra={"dir": str(aligned_dir)})

    # ── 完成 ─────────────────────────────────────────────────────

    logger.info(
        "Pipeline 完成",
        extra={
            "sequence": sequence,
            "total_frames": n_frames,
            "elapsed_s": round(pipeline_elapsed, 2),
            "report": str(html_path),
            "quality_passed": quality_report.passed(),
        },
    )


# ── 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(parse_args())
