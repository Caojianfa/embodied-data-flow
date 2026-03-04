"""训练脚本 — VisionLanguageFusion + ActionPredictor

依赖 main.py 预先生成的 npy 文件（请先运行 main.py）：
  data/output/aligned/{sequence}_visual_feats.npy  — CLIP 视觉特征（冻结）
  data/output/aligned/{sequence}_aligned_imu.npy   — 对齐后 IMU 数据
  data/output/aligned/{sequence}_video_ts.npy      — 视频帧时间戳

训练策略：
  - CLIP 权重冻结，视觉特征直接从 npy 加载，无需 GPU 重跑
  - 语言特征由 CLIP 文本编码器计算一次后固定
  - 可训练参数：VisionLanguageFusion + ActionPredictor
  - 损失函数：MSE（位置差分 + 旋转轴角 + gripper）
  - 优化器：AdamW + CosineAnnealingLR
  - 保存验证集 loss 最低的 checkpoint

用法：
  # 使用默认配置
  python cmd/python/train.py

  # 指定序列和 epoch 数
  python cmd/python/train.py --sequence MH_01_easy --epochs 50

  # 指定设备
  python cmd/python/train.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


# ── 数据集 ─────────────────────────────────────────────────────────

class ActionDataset(Dataset):
    """动作预测训练数据集

    每条样本：(视觉特征, IMU窗口, 动作标签)
    """

    def __init__(
        self,
        visual_feats: np.ndarray,  # (N, 512)
        imu_windows: np.ndarray,   # (N, W, 6)
        labels: np.ndarray,        # (N, horizon, 7)
    ):
        self.visual_feats = torch.from_numpy(visual_feats).float()
        self.imu_windows  = torch.from_numpy(imu_windows).float()
        self.labels       = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.visual_feats[idx], self.imu_windows[idx], self.labels[idx]


# ── 工具函数 ───────────────────────────────────────────────────────

def build_imu_windows(
    aligned_imu: np.ndarray,   # (N, 6)
    valid_indices: np.ndarray, # (M,)
    window_size: int,
) -> np.ndarray:
    """为每个有效帧构建 IMU 历史滑动窗口，shape=(M, W, 6)"""
    windows = []
    for i in valid_indices:
        start  = max(0, i - window_size + 1)
        window = aligned_imu[start : i + 1]
        if len(window) < window_size:
            pad    = np.tile(aligned_imu[0:1], (window_size - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        windows.append(window)
    return np.array(windows, dtype=np.float32)


# ── CLI 参数 ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 VisionLanguageFusion + ActionPredictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config",   default="configs/pipeline.yaml")
    parser.add_argument("--sequence", default=None, help="覆盖配置中的序列名")
    parser.add_argument("--epochs",   type=int, default=None, help="训练轮数（覆盖配置）")
    parser.add_argument("--device",   default=None, help="推理设备 cpu/cuda（覆盖配置）")
    return parser.parse_args()


# ── 训练主函数 ─────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    cfg      = load_config(args.config)
    sequence = args.sequence or cfg.data.sequence
    device   = torch.device(args.device or cfg.model.device)
    train_cfg = cfg.training

    logger.info("训练启动", extra={"sequence": sequence, "device": str(device)})

    # ── 1. 检查并加载预计算 npy 特征 ──────────────────────────────
    aligned_dir = Path(cfg.data.aligned_dir)
    prefix      = aligned_dir / sequence

    required = ["_visual_feats.npy", "_aligned_imu.npy", "_video_ts.npy"]
    for suffix in required:
        path = Path(f"{prefix}{suffix}")
        if not path.exists():
            logger.error(
                "预计算文件不存在，请先运行 main.py 生成特征",
                extra={"missing": str(path), "hint": "python cmd/python/main.py"},
            )
            sys.exit(1)

    visual_feats = np.load(f"{prefix}_visual_feats.npy")  # (N, 512)
    aligned_imu  = np.load(f"{prefix}_aligned_imu.npy")   # (N, 6)
    video_ts     = np.load(f"{prefix}_video_ts.npy")       # (N,)
    N = len(video_ts)

    logger.info("预计算特征加载完成", extra={"n_frames": N, "sequence": sequence})

    # ── 2. 语言特征（CLIP 文本编码，仅运行一次，之后冻结）─────────
    lang_map      = cfg.language.to_dict()
    language_text = lang_map.get(sequence, str(cfg.language.default))

    vision_encoder = VisionEncoder(
        model_name=str(cfg.model.clip_model),
        device=str(device),
    )
    with torch.no_grad():
        text_feat = vision_encoder.encode_text(language_text).to(device).detach()
    del vision_encoder  # 释放 CLIP 占用的内存

    logger.info("语言特征编码完成", extra={"text": language_text, "shape": list(text_feat.shape)})

    # ── 3. 加载 Ground Truth，构造动作标签 ────────────────────────
    sequence_dir  = Path(cfg.data.euroc_base) / sequence
    gt            = GtReader(sequence_dir).load()
    label_builder = LabelBuilder(action_horizon=int(cfg.model.action_horizon))
    valid_indices, labels = label_builder.build(gt, video_ts)
    M = len(valid_indices)

    # ── 4. 构造 IMU 滑动窗口，截取对应视觉特征 ────────────────────
    imu_window_size = int(cfg.model.imu_window)
    imu_windows     = build_imu_windows(aligned_imu, valid_indices, imu_window_size)
    visual_valid    = visual_feats[valid_indices].astype(np.float32)

    logger.info(
        "数据准备完成",
        extra={"valid_frames": M, "label_shape": list(labels.shape)},
    )

    # ── 5. 训练集 / 验证集划分 ────────────────────────────────────
    val_split = float(train_cfg.val_split)
    val_size  = max(1, int(M * val_split))
    rng       = np.random.default_rng(42)
    shuffled  = rng.permutation(M)
    val_idx   = shuffled[:val_size]
    train_idx = shuffled[val_size:]

    batch_size   = int(train_cfg.batch_size)
    train_loader = DataLoader(
        ActionDataset(visual_valid[train_idx], imu_windows[train_idx], labels[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        ActionDataset(visual_valid[val_idx], imu_windows[val_idx], labels[val_idx]),
        batch_size=batch_size, shuffle=False,
    )

    logger.info(
        "数据集划分完成",
        extra={"train": len(train_loader.dataset), "val": len(val_loader.dataset)},
    )

    # ── 6. 模型初始化 ─────────────────────────────────────────────
    vl_fusion = VisionLanguageFusion(dim=int(cfg.model.vision_dim)).to(device)
    predictor = ActionPredictor(
        fusion_dim=int(cfg.model.vision_dim),
        imu_window=imu_window_size,
        action_dim=int(cfg.model.action_dim),
        action_horizon=int(cfg.model.action_horizon),
        d_model=int(cfg.model.transformer_dim),
        nhead=int(cfg.model.transformer_heads),
        num_layers=int(cfg.model.transformer_layers),
    ).to(device)

    n_params = sum(
        p.numel()
        for p in [*vl_fusion.parameters(), *predictor.parameters()]
        if p.requires_grad
    )
    logger.info("模型初始化完成", extra={"trainable_params": n_params})

    # ── 7. 优化器 & 调度器 ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [*vl_fusion.parameters(), *predictor.parameters()],
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    epochs    = args.epochs or int(train_cfg.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ── 8. 训练循环 ───────────────────────────────────────────────
    ckpt_dir     = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_interval = int(train_cfg.log_interval)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练阶段
        vl_fusion.train()
        predictor.train()
        train_loss_sum = 0.0

        for step, (vis, imu, label) in enumerate(train_loader, 1):
            vis   = vis.to(device)    # (B, 512)
            imu   = imu.to(device)    # (B, W, 6)
            label = label.to(device)  # (B, horizon, 7)

            fused = vl_fusion(vis, text_feat)  # (B, 512)
            pred  = predictor(fused, imu)      # (B, horizon, 7)，调用 forward

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*vl_fusion.parameters(), *predictor.parameters()],
                max_norm=1.0,
            )
            optimizer.step()
            train_loss_sum += loss.item()

            if step % log_interval == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}  step {step}/{len(train_loader)}"
                    f"  train_loss={train_loss_sum / step:.6f}"
                )

        scheduler.step()
        train_loss_avg = train_loss_sum / len(train_loader)

        # 验证阶段
        vl_fusion.eval()
        predictor.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for vis, imu, label in val_loader:
                vis   = vis.to(device)
                imu   = imu.to(device)
                label = label.to(device)
                fused = vl_fusion(vis, text_feat)
                pred  = predictor(fused, imu)
                val_loss_sum += criterion(pred, label).item()

        val_loss_avg = val_loss_sum / len(val_loader)
        elapsed      = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{epochs} 完成",
            extra={
                "train_loss": round(train_loss_avg, 6),
                "val_loss":   round(val_loss_avg, 6),
                "lr":         round(scheduler.get_last_lr()[0], 7),
                "elapsed_s":  round(elapsed, 1),
            },
        )

        # 保存验证集最优 checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            ckpt_path     = ckpt_dir / f"{sequence}_best.pt"
            torch.save(
                {
                    "epoch":            epoch,
                    "val_loss":         best_val_loss,
                    "vl_fusion":        vl_fusion.state_dict(),
                    "action_predictor": predictor.state_dict(),
                    "config": {
                        "vision_dim":        int(cfg.model.vision_dim),
                        "action_dim":        int(cfg.model.action_dim),
                        "action_horizon":    int(cfg.model.action_horizon),
                        "imu_window":        imu_window_size,
                        "transformer_dim":   int(cfg.model.transformer_dim),
                        "transformer_heads": int(cfg.model.transformer_heads),
                        "transformer_layers": int(cfg.model.transformer_layers),
                    },
                },
                ckpt_path,
            )
            logger.info(
                "最优模型已保存",
                extra={"path": str(ckpt_path), "val_loss": round(best_val_loss, 6)},
            )

    logger.info(
        "训练完成",
        extra={
            "best_val_loss": round(best_val_loss, 6),
            "checkpoint":    str(ckpt_dir / f"{sequence}_best.pt"),
        },
    )


# ── 入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training(parse_args())
