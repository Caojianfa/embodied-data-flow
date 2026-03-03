# Embodied Data Flow

> 具身智能多模态数据处理与动作预测平台

基于 EuRoC MAV 真实数据集，构建从原始传感器数据到动作预测的完整 Pipeline：

```
PNG 图像序列（20fps）+ IMU CSV（200Hz）
        ↓
  时间戳对齐（线性插值，亚毫秒精度）
        ↓
  CLIP ViT-B/32 视觉编码（512 维）
        ↓
  Cross-Attention 视觉-语言融合
        ↓
  Transformer 动作预测（7D × 10 步）
        ↓
  数据质量评估 + HTML 报告
```

---

## 目录结构

```
embodied-data-flow/
├── configs/
│   └── pipeline.yaml              # 全局配置（路径、模型参数、阈值）
├── data/
│   ├── raw/euroc/                 # EuRoC 数据集（下载后存放）
│   └── output/
│       ├── aligned/               # 对齐后的 npy 文件
│       └── reports/               # HTML 质量报告 + 图表
├── docs/
│   └── tech_solution.md           # 技术方案文档
├── cmd/python/
│   └── main.py                    # Pipeline 入口
├── pkg/
│   ├── ingestion/
│   │   ├── video_reader.py        # EuRoC PNG 序列流式读取
│   │   └── imu_reader.py          # EuRoC IMU CSV 解析
│   ├── processing/
│   │   ├── timestamp_sync.py      # 时间戳对齐（线性插值 O(N+M)）
│   │   └── frame_processor.py     # 帧预处理（resize、归一化、关键帧）
│   ├── models/
│   │   ├── vision_encoder.py      # CLIP ViT-B/32 视觉编码器
│   │   ├── vl_fusion.py           # Cross-Attention VL 融合模块
│   │   └── action_predictor.py    # Transformer 动作预测器
│   ├── quality/
│   │   ├── metrics.py             # 质量指标计算
│   │   └── reporter.py            # HTML 报告 + 图表生成
│   └── utils/
│       ├── logger.py              # 结构化 JSON 日志
│       └── config.py              # YAML 配置加载
└── scripts/
    └── download_data.sh           # EuRoC 数据集下载脚本
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. 下载数据集

```bash
# 前台运行（约 3.4GB，MH_01_easy + MH_02_easy）
bash scripts/download_data.sh

# 后台下载（可关闭终端）
nohup bash scripts/download_data.sh > download.log 2>&1 &
tail -f download.log   # 实时查看进度
```

下载完成后目录结构：

```
data/raw/euroc/
├── MH_01_easy/mav0/
│   ├── cam0/data/          # PNG 图像序列（~3682 张）
│   ├── cam0/data.csv       # 图像时间戳（纳秒）
│   ├── imu0/data.csv       # IMU 数据（200Hz）
│   └── state_groundtruth_estimate0/data.csv  # 地面真值位姿+速度
└── MH_02_easy/mav0/        # 结构相同
```

### 3. 运行 Pipeline

```bash
# 处理 MH_01_easy 全部帧（首次运行会下载 CLIP 模型 ~300MB）
python cmd/python/main.py

# 快速调试：只处理前 200 帧
python cmd/python/main.py --max-frames 200

# 切换序列
python cmd/python/main.py --sequence MH_02_easy

# 对比最近邻对齐 vs 线性插值
python cmd/python/main.py --sync-method nearest
python cmd/python/main.py --sync-method linear
```

### 4. 查看结果

```
data/output/reports/
├── report_MH_01_easy.html          # 用浏览器直接打开
├── summary_MH_01_easy.json         # JSON 格式质量摘要
└── img/
    ├── sync_error.png              # 时间戳同步误差分布
    ├── imu_accel.png               # 加速度时序图
    ├── imu_gyro.png                # 角速度时序图
    └── blur.png                    # 视频模糊度逐帧曲线

data/output/aligned/
├── MH_01_easy_aligned_imu.npy     # 对齐后 IMU，shape=(N, 6)
├── MH_01_easy_visual_feats.npy    # 视觉特征，shape=(N, 512)
├── MH_01_easy_fused_feats.npy     # 融合特征，shape=(N, 512)
├── MH_01_easy_actions.npy         # 动作序列，shape=(N, 10, 7)
└── MH_01_easy_video_ts.npy        # 视频帧时间戳，shape=(N,)
```

---

## 技术架构详解

### 时间戳对齐（核心模块）

**挑战**：视频 20fps（帧间隔 50ms）vs IMU 200Hz（采样间隔 5ms），采样率差 10×

**算法**：O(N+M) 双指针线性插值

```
IMU 时间轴:  t0 ─── t1 ─── t2 ─── t3
视频帧时间:       v1        v2

对于 v1：找到 t1 < v1 < t2
  插值权重 w = (v1 - t1) / (t2 - t1)
  结果 = (1-w)·imu[t1] + w·imu[t2]
  同步误差 = min(v1-t1, t2-v1)
```

期望指标：同步误差均值 < 5ms，标准差 < 3ms

### 视觉-语言融合

```
视觉特征 (512) ── Q ─┐
                      ├→ Cross-Attention → LayerNorm → FFN → 融合特征 (512)
语言特征 (512) ── KV ─┘
```

语言输入使用 EuRoC 序列对应的场景描述，在 `configs/pipeline.yaml` 的 `language` 节配置。

### 动作预测

**Action Chunking**：一次预测未来 T=10 步动作，减少推理次数，输出更平滑。

```
输入 Token 序列：
  Token 0:      VL 融合特征（场景理解）
  Token 1..10:  IMU 历史窗口（最近 10 个对齐采样，运动状态）

Transformer Encoder (4层, 8头, dim=256)

输出：(N, 10, 7)
  action[:, :, 0:3] = delta_position (x, y, z)
  action[:, :, 3:6] = delta_rotation (rx, ry, rz)
  action[:, :, 6]   = gripper_openness ∈ [0, 1]
```

> **注意**：本项目使用随机初始化权重进行架构验证（Pipeline 端到端连通性），
> 动作预测结果仅供格式验证，非真实策略输出。

### 数据质量指标

| 指标 | 期望值 |
|------|--------|
| 帧完整性 | > 99% |
| IMU 完整性 | > 99.5% |
| 同步误差均值 | < 5 ms |
| 同步误差标准差 | < 3 ms |
| IMU 异常值率 | < 0.1% |

---

## 配置说明

`configs/pipeline.yaml` 关键配置项：

```yaml
data:
  sequence: "MH_01_easy"      # 当前处理的序列

pipeline:
  max_frames: null             # null = 全部帧；整数 = 限制帧数（调试）
  batch_size: 32               # CLIP 编码批量大小

model:
  device: "cpu"                # 有 GPU 时改为 "cuda"
  clip_model: "ViT-B/32"

language:
  MH_01_easy: "Flying over Machine Hall, approach the corridor"
  # 可为每个序列配置不同的场景描述
```

---

## 数据集

**EuRoC MAV Dataset**（ETH Zurich ASL）

| 属性 | 值 |
|------|----|
| 相机 | 立体相机，20fps，752×480，PNG 格式 |
| IMU | 200Hz，[wx,wy,wz,ax,ay,az] |
| 地面真值 | 6DoF 位姿 + 速度（MoCap/Leica 测量） |
| 使用序列 | MH_01_easy（1.9GB）+ MH_02_easy（1.5GB） |
| 论文 | Burri et al., IJRR 2017 |

---

## 扩展方向

- **数据摄取层** → Kafka Topic（VideoReader 变为 Consumer，支持多路并发）
- **处理引擎** → Flink 流式处理（保留分层接口，直接替换）
- **模型推理** → GPU 加速 + TorchScript/ONNX 导出
- **动作预测** → 引入遥操作数据集（BridgeData V2 等）进行真实训练
