# Embodied Data Flow

> 具身智能多模态数据处理与动作预测平台

基于 EuRoC MAV 真实数据集，构建从原始传感器数据到动作预测的**完整闭环 Pipeline**：

```
PNG 图像序列（20fps）+ IMU CSV（200Hz）+ Ground Truth 位姿
        ↓
  时间戳对齐（线性插值，亚毫秒精度）
        ↓
  CLIP ViT-B/32 视觉编码（512 维，冻结）
        ↓
  Cross-Attention 视觉-语言融合（可训练）
        ↓
  Transformer 动作预测（7D × 10 步，可训练）
        ↓
  监督训练（Ground Truth 差分标签）+ 评估对比
        ↓
  数据质量评估 + HTML 报告
```

---

## 目录结构

```
embodied-data-flow/
├── configs/
│   └── pipeline.yaml              # 全局配置（路径、模型参数、训练超参、质量阈值）
├── data/
│   ├── raw/euroc/                 # EuRoC 数据集（下载后存放）
│   └── output/
│       ├── aligned/               # 对齐后的 npy 中间文件
│       ├── checkpoints/           # 训练模型权重（.pt）
│       └── reports/               # 质量报告 + 评估图表
├── docs/
│   └── tech_solution.md           # 技术方案文档
├── cmd/python/
│   ├── main.py                    # Pipeline 入口（数据处理 + 推理）
│   ├── train.py                   # 训练脚本（VLFusion + ActionPredictor）
│   └── evaluate.py                # 评估脚本（推理结果 vs Ground Truth）
├── pkg/
│   ├── ingestion/
│   │   ├── video_reader.py        # EuRoC PNG 序列流式读取
│   │   ├── imu_reader.py          # EuRoC IMU CSV 解析
│   │   └── gt_reader.py           # EuRoC Ground Truth 位姿读取
│   ├── processing/
│   │   ├── timestamp_sync.py      # 时间戳对齐（线性插值 O(N+M)）
│   │   ├── frame_processor.py     # 帧预处理（resize、归一化、关键帧）
│   │   └── label_builder.py       # Ground Truth → 动作标签（差分 + 轴角）
│   ├── models/
│   │   ├── vision_encoder.py      # CLIP ViT-B/32 视觉编码器
│   │   ├── vl_fusion.py           # Cross-Attention VL 融合模块
│   │   └── action_predictor.py    # Transformer 动作预测器（ACT 简化版）
│   ├── quality/
│   │   ├── metrics.py             # 质量指标计算（滑动窗口 Nσ 异常值检测）
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

# 后台下载
nohup bash scripts/download_data.sh > download.log 2>&1 &
tail -f download.log
```

### 3. 运行 Pipeline（生成特征文件）

```bash
# 处理 MH_01_easy 全部帧（首次运行会下载 CLIP 模型 ~300MB）
python cmd/python/main.py

# 调试：只处理前 200 帧
python cmd/python/main.py --max-frames 200

# 切换序列
python cmd/python/main.py --sequence MH_02_easy

# 加载训练好的权重进行推理
python cmd/python/main.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt
```

### 4. 训练模型

```bash
# 使用默认配置训练（需先完成步骤 3 生成 npy 特征文件）
python cmd/python/train.py

# 自定义参数
python cmd/python/train.py --sequence MH_01_easy --epochs 50 --device cuda
```

训练过程输出（每 20 个 step 打印一次）：
```
Epoch 1/30 step 20/52  train_loss=0.000821
Epoch 1/30 完成  train_loss=0.000734  val_loss=0.000698  lr=0.0003000
最优模型已保存  path=data/output/checkpoints/MH_01_easy_best.pt  val_loss=0.000698
```

### 5. 评估推理质量

```bash
python cmd/python/evaluate.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt
```

控制台输出：
```
==================================================
序列：MH_01_easy   Checkpoint epoch 28
  位置差分 MAE : 12.34 mm
  旋转差分 MAE :  3.56 mrad

  各维度 MAE：
    dx(mm)          8.1200
    dy(mm)         10.4500
    dz(mm)         18.3100
    rx(mrad)        2.8900
    ry(mrad)        3.1200
    rz(mrad)        4.7800
    gripper         0.0000

  各预测步 MAE（步 1 → 10）：
    步  1   8.2000  ████████
    步  2   9.1000  █████████
    步 10  18.5000  ████████████████████
==================================================
```

---

## 输出文件

```
data/output/
├── aligned/
│   ├── MH_01_easy_visual_feats.npy    # CLIP 视觉特征，shape=(N, 512)
│   ├── MH_01_easy_aligned_imu.npy     # 对齐后 IMU，shape=(N, 6)
│   ├── MH_01_easy_fused_feats.npy     # VL 融合特征，shape=(N, 512)
│   ├── MH_01_easy_actions.npy         # 预测动作，shape=(N, 10, 7)
│   └── MH_01_easy_video_ts.npy        # 视频帧时间戳，shape=(N,)
├── checkpoints/
│   └── MH_01_easy_best.pt             # 验证集最优模型权重
└── reports/
    ├── report_MH_01_easy.html          # 数据质量报告（浏览器打开）
    ├── summary_MH_01_easy.json         # 质量摘要 JSON
    ├── evaluation_MH_01_easy.json      # 动作预测评估指标 JSON
    └── img/
        ├── sync_error.png              # 时间戳同步误差分布
        ├── imu_accel.png               # 加速度时序图
        ├── imu_gyro.png                # 角速度时序图
        ├── blur.png                    # 视频模糊度逐帧曲线
        ├── eval_horizon_error.png      # 各预测步误差曲线
        └── eval_action_compare.png     # GT vs 预测位置差分对比
```

---

## 技术架构

### 完整数据闭环

```
┌──────────────────────────────────────────────────────┐
│                   数据摄取层（Ingestion）               │
│  VideoReader（流式PNG）  ImuReader  GtReader（位姿真值）│
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│                   处理层（Processing）                  │
│  TimestampSync（O(N+M)双指针线性插值）                  │
│  FrameProcessor（resize/归一化/关键帧检测）             │
│  LabelBuilder（GT差分 → 位置+轴角动作标签）             │
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│                   模型层（Models）                      │
│  VisionEncoder: CLIP ViT-B/32 → (N, 512)  【冻结】   │
│  VisionLanguageFusion: Cross-Attention    【可训练】  │
│  ActionPredictor: Transformer Encoder     【可训练】  │
│    输出: (N, 10, 7) = 10步 × [dx,dy,dz,rx,ry,rz,gripper]│
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│               训练 & 评估（Train / Evaluate）          │
│  Loss: MSE(预测动作, GT差分标签)                       │
│  优化器: AdamW + CosineAnnealingLR                    │
│  评估指标: 位置MAE(mm) + 旋转MAE(mrad) + 各步误差曲线  │
└──────────────────────────────────────────────────────┘
```

### 时间戳对齐

**核心算法**：O(N+M) 双指针线性插值，将 IMU（200Hz）对齐到视频帧（20fps）

```
IMU（200Hz）: ● ● ● ● ● ● ● ● ● ● ●   每 5ms 一个采样
视频（20fps）:↑                   ↑     每 50ms 一帧

对于每个视频帧时刻 v，找到满足 t₀ ≤ v ≤ t₁ 的相邻 IMU 对：
  w = (v - t₀) / (t₁ - t₀)
  imu_aligned = (1-w)·imu[t₀] + w·imu[t₁]
```

### Ground Truth 动作标签构造

```
GT数据（200Hz）: position(3) + quaternion(4) + velocity(3)
        ↓ 插值对齐到视频帧时间戳（20fps）
        ↓ 差分计算单步动作
  delta_pos[t] = pos[t+1] - pos[t]              → 3维，单位 m
  delta_rot[t] = q[t]⁻¹ ⊗ q[t+1] → 轴角        → 3维，单位 rad
  gripper[t]   = 0.0                            → 1维
        ↓ 构造 Action Chunking 窗口
  label[i] = [delta[i], delta[i+1], ..., delta[i+9]]  shape=(10, 7)
```

### IMU 异常值检测（滑动窗口 Nσ）

```
全局 3σ（旧）: 用全局均值统计，把正常飞行机动误判为异常 → 6.3% 误报
滑动窗口 4.5σ（新）: 2s 窗口局部统计，只捕捉真正的传感器硬件毛刺 → 0.07%
```

---

## 配置说明

`configs/pipeline.yaml` 关键配置项：

```yaml
model:
  device: "cpu"          # 有 GPU 时改为 "cuda"
  clip_model: "ViT-B/32"
  action_horizon: 10     # 预测未来步数
  imu_window: 10         # IMU 历史窗口大小

training:
  epochs: 30
  batch_size: 64
  lr: 3e-4
  val_split: 0.1         # 验证集比例

quality:
  max_imu_outlier_rate: 0.001    # IMU 异常值率阈值
  imu_outlier_window_s: 2.0      # 滑动窗口大小（秒）
  imu_outlier_sigma: 4.5         # σ 倍数
```

---

## 数据集

**EuRoC MAV Dataset**（ETH Zurich ASL）

| 属性 | 值 |
|------|----|
| 相机 | 立体相机，20fps，752×480，PNG 格式 |
| IMU | 200Hz，[wx,wy,wz,ax,ay,az] |
| Ground Truth | 6DoF 位姿 + 速度（Leica 激光追踪，200Hz） |
| 使用序列 | MH_01_easy（1.9GB）+ MH_02_easy（1.5GB）|
| 论文 | Burri et al., IJRR 2017 |

---

## 扩展方向

- **数据摄取层** → Kafka Topic（VideoReader 变为 Consumer，支持多路并发）
- **处理引擎** → Flink 流式处理（保留分层接口，直接替换）
- **模型推理** → GPU 加速 + TorchScript/ONNX 导出
- **训练数据** → 引入遥操作数据集（BridgeData V2、Open X-Embodiment）替换伪标签
