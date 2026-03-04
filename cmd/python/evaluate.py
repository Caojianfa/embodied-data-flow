"""评估脚本 — 对比推理动作序列与 Ground Truth

依赖 main.py 预先生成的 npy 文件 + train.py 训练出的 checkpoint：
  data/output/aligned/{sequence}_visual_feats.npy
  data/output/aligned/{sequence}_aligned_imu.npy
  data/output/aligned/{sequence}_video_ts.npy
  data/output/checkpoints/{sequence}_best.pt

输出到 data/output/reports/：
  evaluation_{sequence}.json        — 指标摘要
  img/eval_horizon_error.png        — 各预测步误差曲线（误差如何随预测步增大）
  img/eval_action_compare.png       — GT vs 预测的位置差分对比（前 300 帧）

核心指标：
  pos_mae_mm   — 位置差分 MAE，单位 mm（越小越好）
  rot_mae_mrad — 旋转差分 MAE，单位 mrad（越小越好）
  per_step_mae — 各预测步的平均误差（验证 Action Chunking 的质量）

用法：
  python cmd/python/evaluate.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt
  python cmd/python/evaluate.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt --sequence MH_01_easy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.ingestion.gt_reader import GtReader
from pkg.models.action_predictor import ActionPredictor
from pkg.models.vision_encoder import VisionEncoder
from pkg.models.vl_fusion import VisionLanguageFusion
from pkg.processing.label_builder import LabelBuilder
from pkg.utils.config import load_config
from pkg.utils.logger import get_logger

logger = get_logger(__name__)

DIM_NAMES = ["dx(mm)", "dy(mm)", "dz(mm)", "rx(mrad)", "ry(mrad)", "rz(mrad)", "gripper"]


# ── CLI 参数 ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估动作预测结果 vs Ground Truth")
    parser.add_argument("--config",     default="configs/pipeline.yaml")
    parser.add_argument("--sequence",   default=None)
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--device",     default=None)
    return parser.parse_args()


# ── 推理 ──────────────────────────────────────────────────────────

def run_inference(
    vl_fusion: VisionLanguageFusion,
    predictor: ActionPredictor,
    visual_feats: np.ndarray,   # (M, 512)
    imu_windows: np.ndarray,    # (M, W, 6)
    text_feat: torch.Tensor,    # (512,)
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """批量推理，返回预测动作 shape=(M, horizon, 7)"""
    vl_fusion.eval()
    predictor.eval()
    M = len(visual_feats)
    all_preds = []

    with torch.no_grad():
        for start in range(0, M, batch_size):
            vis = torch.from_numpy(visual_feats[start : start + batch_size]).float().to(device)
            imu = torch.from_numpy(imu_windows[start : start + batch_size]).float().to(device)
            fused = vl_fusion(vis, text_feat)
            pred  = predictor(fused, imu)           # (B, horizon, 7)
            all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)        # (M, horizon, 7)


# ── 指标计算 ──────────────────────────────────────────────────────

def compute_metrics(
    preds: np.ndarray,   # (M, horizon, 7)
    labels: np.ndarray,  # (M, horizon, 7)
) -> dict:
    """
    计算多维度 MAE 指标

    Returns:
        {
          "overall_pos_mae_mm":   float,  位置差分 MAE（mm）
          "overall_rot_mae_mrad": float,  旋转差分 MAE（mrad）
          "per_step_mae":         list,   各预测步的平均 MAE（7维均值）
          "per_dim_mae":          dict,   每个动作维度的 MAE（含单位换算）
        }
    """
    err = np.abs(preds - labels)              # (M, horizon, 7)

    # 位置 / 旋转单位换算（m → mm，rad → mrad）
    err_display        = err.copy()
    err_display[..., :3]  *= 1000.0           # m → mm
    err_display[..., 3:6] *= 1000.0           # rad → mrad

    # 整体位置 / 旋转 MAE
    overall_pos_mae  = float(err_display[..., :3].mean())
    overall_rot_mae  = float(err_display[..., 3:6].mean())

    # 各预测步 MAE（对帧数和7个维度取均值，但位置/旋转分别换算后平均）
    # 这里统一用换算后的量反映"平均绝对误差"
    per_step_mae = err_display.mean(axis=(0, 2)).tolist()   # (horizon,)

    # 每个动作维度的 MAE
    per_dim_raw = err_display.mean(axis=(0, 1))             # (7,)
    per_dim_mae = {name: round(float(v), 4) for name, v in zip(DIM_NAMES, per_dim_raw)}

    return {
        "overall_pos_mae_mm":   round(overall_pos_mae, 4),
        "overall_rot_mae_mrad": round(overall_rot_mae, 4),
        "per_step_mae":         [round(v, 4) for v in per_step_mae],
        "per_dim_mae":          per_dim_mae,
    }


# ── 可视化 ────────────────────────────────────────────────────────

def plot_horizon_error(
    per_step_mae: list,
    img_dir: Path,
    sequence: str,
) -> None:
    """各预测步 MAE 曲线：误差如何随预测步增大"""
    horizon = len(per_step_mae)
    steps   = list(range(1, horizon + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, per_step_mae, marker="o", linewidth=2, color="#1976D2")
    ax.fill_between(steps, per_step_mae, alpha=0.15, color="#1976D2")
    ax.set_xlabel("预测步（Action Horizon）", fontsize=11)
    ax.set_ylabel("平均 MAE（位置 mm / 旋转 mrad）", fontsize=11)
    ax.set_title(f"{sequence} — 各预测步误差", fontsize=12)
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(img_dir / "eval_horizon_error.png", dpi=120)
    plt.close(fig)


def plot_action_compare(
    preds: np.ndarray,   # (M, horizon, 7)
    labels: np.ndarray,  # (M, horizon, 7)
    img_dir: Path,
    sequence: str,
    n_frames: int = 300,
) -> None:
    """GT vs 预测的位置差分对比（取第 1 步预测，前 n_frames 帧）"""
    n    = min(n_frames, len(preds))
    axes = ["dx", "dy", "dz"]
    colors_gt   = ["#E53935", "#43A047", "#1E88E5"]
    colors_pred = ["#EF9A9A", "#A5D6A7", "#90CAF9"]

    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for k, (ax, name, cg, cp) in enumerate(zip(axs, axes, colors_gt, colors_pred)):
        gt_vals   = labels[:n, 0, k] * 1000   # m → mm
        pred_vals = preds[:n,  0, k] * 1000
        ax.plot(gt_vals,   label="GT",   color=cg, linewidth=1.2)
        ax.plot(pred_vals, label="预测", color=cp, linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"{name} (mm)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    axs[0].set_title(f"{sequence} — 位置差分对比（第 1 步，前 {n} 帧）", fontsize=12)
    axs[-1].set_xlabel("帧索引", fontsize=11)
    plt.tight_layout()
    fig.savefig(img_dir / "eval_action_compare.png", dpi=120)
    plt.close(fig)


# ── 主函数 ────────────────────────────────────────────────────────

def run_evaluate(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    sequence = args.sequence or cfg.data.sequence
    device   = torch.device(args.device or cfg.model.device)

    logger.info("评估启动", extra={"sequence": sequence, "checkpoint": args.checkpoint})

    # ── 1. 检查并加载预计算特征 ───────────────────────────────────
    aligned_dir = Path(cfg.data.aligned_dir)
    prefix      = aligned_dir / sequence

    for suffix in ["_visual_feats.npy", "_aligned_imu.npy", "_video_ts.npy"]:
        if not Path(f"{prefix}{suffix}").exists():
            logger.error(
                "预计算文件不存在，请先运行 main.py",
                extra={"missing": f"{prefix}{suffix}"},
            )
            sys.exit(1)

    visual_feats = np.load(f"{prefix}_visual_feats.npy")   # (N, 512)
    aligned_imu  = np.load(f"{prefix}_aligned_imu.npy")    # (N, 6)
    video_ts     = np.load(f"{prefix}_video_ts.npy")        # (N,)

    # ── 2. 语言特征（CLIP 文本编码，仅一次）─────────────────────
    lang_map      = cfg.language.to_dict()
    language_text = lang_map.get(sequence, str(cfg.language.default))
    vision_encoder = VisionEncoder(model_name=str(cfg.model.clip_model), device=str(device))
    with torch.no_grad():
        text_feat = vision_encoder.encode_text(language_text).to(device).detach()
    del vision_encoder

    # ── 3. GT 标签构造 ────────────────────────────────────────────
    sequence_dir  = Path(cfg.data.euroc_base) / sequence
    gt            = GtReader(sequence_dir).load()
    label_builder = LabelBuilder(action_horizon=int(cfg.model.action_horizon))
    valid_indices, labels = label_builder.build(gt, video_ts)   # (M, horizon, 7)

    # ── 4. 构造 IMU 窗口 ──────────────────────────────────────────
    W = int(cfg.model.imu_window)
    imu_windows = []
    for i in valid_indices:
        start  = max(0, i - W + 1)
        window = aligned_imu[start : i + 1]
        if len(window) < W:
            pad    = np.tile(aligned_imu[0:1], (W - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        imu_windows.append(window)
    imu_windows  = np.array(imu_windows, dtype=np.float32)
    visual_valid = visual_feats[valid_indices].astype(np.float32)

    # ── 5. 加载 checkpoint，初始化模型 ───────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("checkpoint 文件不存在", extra={"path": str(ckpt_path)})
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)

    vl_fusion = VisionLanguageFusion(dim=int(cfg.model.vision_dim)).to(device)
    predictor = ActionPredictor(
        fusion_dim=int(cfg.model.vision_dim),
        imu_window=W,
        action_dim=int(cfg.model.action_dim),
        action_horizon=int(cfg.model.action_horizon),
        d_model=int(cfg.model.transformer_dim),
        nhead=int(cfg.model.transformer_heads),
        num_layers=int(cfg.model.transformer_layers),
    ).to(device)

    vl_fusion.load_state_dict(ckpt["vl_fusion"])
    predictor.load_state_dict(ckpt["action_predictor"])

    logger.info(
        "模型权重加载完成",
        extra={"epoch": ckpt.get("epoch"), "train_val_loss": ckpt.get("val_loss")},
    )

    # ── 6. 批量推理 ───────────────────────────────────────────────
    preds = run_inference(
        vl_fusion, predictor,
        visual_valid, imu_windows, text_feat,
        device, batch_size=int(cfg.pipeline.batch_size),
    )
    logger.info("推理完成", extra={"preds_shape": list(preds.shape)})

    # ── 7. 计算指标 ───────────────────────────────────────────────
    metrics = compute_metrics(preds, labels)

    logger.info(
        "评估指标",
        extra={
            "pos_mae_mm":   metrics["overall_pos_mae_mm"],
            "rot_mae_mrad": metrics["overall_rot_mae_mrad"],
            "per_step_mae": metrics["per_step_mae"],
        },
    )

    # ── 8. 保存结果 ───────────────────────────────────────────────
    reports_dir = Path(cfg.data.reports_dir)
    img_dir     = reports_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # JSON 摘要
    summary = {
        "sequence":    sequence,
        "checkpoint":  str(ckpt_path),
        "epoch":       ckpt.get("epoch"),
        "val_loss":    ckpt.get("val_loss"),
        "n_frames":    int(len(valid_indices)),
        "horizon":     int(cfg.model.action_horizon),
        **metrics,
    }
    json_path = reports_dir / f"evaluation_{sequence}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # 可视化
    plot_horizon_error(metrics["per_step_mae"], img_dir, sequence)
    plot_action_compare(preds, labels, img_dir, sequence)

    logger.info(
        "评估完成",
        extra={
            "json":  str(json_path),
            "plots": str(img_dir),
            "pos_mae_mm":   metrics["overall_pos_mae_mm"],
            "rot_mae_mrad": metrics["overall_rot_mae_mrad"],
        },
    )

    # 控制台汇总
    print("\n" + "=" * 50)
    print(f"序列：{sequence}   Checkpoint epoch {ckpt.get('epoch')}")
    print(f"  位置差分 MAE : {metrics['overall_pos_mae_mm']:.2f} mm")
    print(f"  旋转差分 MAE : {metrics['overall_rot_mae_mrad']:.2f} mrad")
    print("\n  各维度 MAE：")
    for name, val in metrics["per_dim_mae"].items():
        print(f"    {name:<14} {val:.4f}")
    print("\n  各预测步 MAE（步 1 → 10）：")
    for step, val in enumerate(metrics["per_step_mae"], 1):
        bar = "█" * int(val / max(metrics["per_step_mae"]) * 20)
        print(f"    步 {step:2d}  {val:7.4f}  {bar}")
    print("=" * 50)


# ── 入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluate(parse_args())
