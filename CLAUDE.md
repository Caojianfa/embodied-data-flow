# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Embodied Data Flow** 是一个具身智能多模态数据处理与动作预测平台，构建从真实传感器数据（EuRoC MAV Dataset）到动作预测的端到端 Pipeline。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# 下载数据集（EuRoC MH_01_easy + MH_02_easy）
bash scripts/download_data.sh
# 后台下载
nohup bash scripts/download_data.sh > download.log 2>&1 &

# 快速测试（前 100 帧，首次运行用此验证环境）
python cmd/python/main.py --max-frames 100

# 完整运行（首次会下载 CLIP 模型 ~300MB）
python cmd/python/main.py

# 切换序列
python cmd/python/main.py --sequence MH_02_easy --max-frames 200

# 切换对齐方法（linear 或 nearest）
python cmd/python/main.py --sync-method nearest

# 不保存 npy 中间文件
python cmd/python/main.py --no-save
```

## 项目架构

项目采用**分层数据流**架构，数据从传感器经过摄取、处理、模型推理到质量评估：

```
cmd/python/main.py          # Pipeline 入口 + CLI 参数
configs/pipeline.yaml       # 全局配置（采样率、模型参数、阈值）
pkg/
├── ingestion/              # 数据摄取层（流式读取，避免 OOM）
│   ├── video_reader.py     # EuRoC cam0 PNG 序列 + 时间戳解析
│   └── imu_reader.py       # EuRoC imu0 CSV（200Hz，6 维）
├── processing/             # 处理引擎
│   ├── timestamp_sync.py   # 核心：O(N+M) 时间戳对齐算法
│   └── frame_processor.py  # Resize/归一化/模糊度检测/关键帧提取
├── models/                 # AI 模型层
│   ├── vision_encoder.py   # CLIP ViT-B/32（512 维特征）
│   ├── vl_fusion.py        # Cross-Attention VL 融合
│   └── action_predictor.py # Transformer Encoder 动作预测（10 步）
├── quality/                # 质量评估层
│   ├── metrics.py          # 完整性/同步误差/模糊度/IMU 异常值指标
│   └── reporter.py         # HTML 报告 + PNG 图表 + JSON 摘要
└── utils/
    ├── config.py           # YAML 配置加载（支持点号属性访问）
    └── logger.py           # 结构化 JSON 日志（LoggerMixin）
```

## 核心算法

**时间戳对齐**（`pkg/processing/timestamp_sync.py`）是项目核心数据工程能力：
- 将 IMU（200Hz）对齐到视频帧（20fps），采样率差 10×
- 线性插值（推荐）：`w = (v - t0) / (t1 - t0); result = (1-w)*imu[j] + w*imu[j+1]`
- 最近邻：找最近 IMU 时间戳
- 复杂度 O(N+M)，双指针实现

**模型流水线**：
1. CLIP ViT-B/32 → (B, 512) 视觉特征
2. Cross-Attention 融合视觉 + 文本描述 → (B, 512) 融合特征
3. Transformer Encoder（4 层，8 头，256 维）预测 → (B, 10, 7) 动作序列

动作空间：`[delta_x, delta_y, delta_z, rx, ry, rz, gripper_openness]`

## 输出结构

```
data/output/
├── reports/
│   ├── report_{sequence}.html     # 质量报告（浏览器打开）
│   ├── summary_{sequence}.json    # JSON 质量摘要
│   └── img/                       # 同步误差/IMU/模糊度图表
└── aligned/
    ├── {sequence}_aligned_imu.npy
    ├── {sequence}_visual_feats.npy
    ├── {sequence}_fused_feats.npy
    ├── {sequence}_actions.npy
    └── {sequence}_video_ts.npy
```

## 配置说明

关键配置项在 `configs/pipeline.yaml`：
- `data.sequence`：当前处理序列（MH_01_easy / MH_02_easy）
- `model.device`：`"cpu"` 或 `"cuda"`（有 GPU 时改为 cuda）
- `model.clip_model`：`"ViT-B/32"`（首次运行自动下载）
- `pipeline.max_frames`：`null` 表示全部帧
- `language`：各序列对应的文本描述（用于 VL 融合）

## 数据集格式

EuRoC MAV Dataset 目录结构：
```
data/raw/euroc/{sequence}/mav0/
├── cam0/data.csv       # 时间戳列表（纳秒）+ data/ PNG 图像
└── imu0/data.csv       # 6 维 IMU 数据（纳秒时间戳）
```

## 质量指标期望值

| 指标 | 期望 |
|------|------|
| 帧完整性 | > 99% |
| 同步误差均值 | < 5ms |
| IMU 异常值率 | < 0.1% |
| 处理吞吐量 | > 100 fps |
