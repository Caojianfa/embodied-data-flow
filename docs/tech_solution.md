# Embodied Data Flow — 技术方案

> 具身智能多模态数据处理与动作预测平台

---

## 一、项目定位

### 1.1 背景与目标

| 维度 | 说明 |
|------|------|
| 项目定位 | **数据工程 + AI 并重**，用工程师视角构建完整数据闭环 |
| 核心目标 | 本地可运行、有实际输出、训练-推理-评估全流程打通 |

### 1.2 展示的关键能力

```
大数据工程能力（迁移自 Flink/Spark 经验）
  → 流式处理架构、高吞吐设计、时序数据对齐

多模态数据处理（新技能，项目中建立 credibility）
  → OpenCV 视频处理、IMU 信号处理、时间戳同步

AI 集成能力（基于 PyTorch/Transformer 经验）
  → 视觉特征提取（CLIP/ViT）
  → 视觉-语言模态融合（Cross-Attention）
  → Transformer 动作预测（ACT 架构）
  → 监督训练闭环（Ground Truth 差分标签）

工程化能力（10 年数据工程背景）
  → 可观测性（结构化日志 + 性能指标）
  → 数据质量评估体系（滑动窗口异常值检测）
  → 配置驱动的可扩展设计
```

---

## 二、整体架构

```
raw data (EuRoC MAV Dataset)
  ├── video  (PNG 图像序列，20fps，含时间戳 CSV)
  ├── imu    (CSV，200Hz，[wx,wy,wz,ax,ay,az]，纳秒时间戳)
  ├── gt     (state_groundtruth_estimate0，200Hz，位置+四元数+速度)
  └── language instruction (序列对应场景描述文本)
          │
          ▼
┌─────────────────────────────────────┐
│           Ingestion Layer           │
│  VideoReader  ImuReader  GtReader   │
│  （流式读取，避免 OOM）               │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│          Processing Engine          │
│                                     │
│  TimestampSync                      │
│  ├─ 线性插值：IMU(200Hz) → 视频帧   │
│  └─ O(N+M) 双指针，亚毫秒级精度     │
│                                     │
│  FrameProcessor                     │
│  ├─ 关键帧检测（场景变化阈值）        │
│  └─ 图像预处理（resize/normalize）   │
│                                     │
│  LabelBuilder（训练专用）            │
│  ├─ GT插值对齐到视频帧时间戳         │
│  ├─ 位置差分 delta_pos (3D, m)      │
│  ├─ 四元数差分→轴角 delta_rot (3D)  │
│  └─ Action Chunking 窗口构造        │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│             AI Models               │
│                                     │
│  VisionEncoder (CLIP ViT-B/32)      │
│  └─ 每帧 → 512 维视觉特征【冻结】   │
│                                     │
│  VisionLanguageFusion               │
│  └─ Cross-Attention 融合 → 512 维   │
│     【训练时可训练】                  │
│                                     │
│  ActionPredictor (Transformer)      │
│  ├─ input: 融合特征 + IMU 窗口      │
│  └─ output: (N, 10, 7) 动作序列     │
│     【训练时可训练】                  │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│        Train / Evaluate Loop        │
│                                     │
│  train.py                           │
│  ├─ MSE Loss(预测动作, GT标签)       │
│  ├─ AdamW + CosineAnnealingLR       │
│  └─ 保存最优 checkpoint (.pt)       │
│                                     │
│  evaluate.py                        │
│  ├─ 位置 MAE (mm)                   │
│  ├─ 旋转 MAE (mrad)                 │
│  ├─ 各预测步误差曲线                  │
│  └─ GT vs 预测对比图                 │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│         Quality & Monitoring        │
│  ├─ 帧/IMU 完整性检查                │
│  ├─ 滑动窗口 Nσ 异常值检测           │
│  ├─ 时间戳同步误差分布               │
│  ├─ 视频模糊度检测（拉普拉斯方差）    │
│  └─ HTML 质量报告输出               │
└─────────────────────────────────────┘
```

---

## 三、目录结构

```
embodied-data-flow/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── pipeline.yaml              # 全局配置（路径、模型参数、训练超参、质量阈值）
│
├── data/
│   ├── raw/euroc/
│   │   ├── MH_01_easy/mav0/
│   │   │   ├── cam0/data/                         # PNG 图像序列
│   │   │   ├── cam0/data.csv                      # 图像时间戳（ns）
│   │   │   ├── imu0/data.csv                      # IMU 数据（200Hz）
│   │   │   └── state_groundtruth_estimate0/data.csv  # GT 位姿+速度（200Hz）
│   │   └── MH_02_easy/                            # 结构同上
│   └── output/
│       ├── aligned/               # 对齐后的 npy 中间文件
│       ├── checkpoints/           # 模型训练权重（.pt）
│       └── reports/               # 质量报告 + 评估图表
│
├── docs/
│   └── tech_solution.md           # 本文档
│
├── cmd/python/
│   ├── main.py                    # Pipeline 入口（数据处理 + 推理）
│   ├── train.py                   # 模型训练脚本
│   └── evaluate.py                # 推理评估脚本
│
├── pkg/
│   ├── ingestion/
│   │   ├── video_reader.py        # EuRoC PNG 序列流式读取
│   │   ├── imu_reader.py          # IMU CSV 解析（纳秒→毫秒）
│   │   └── gt_reader.py           # Ground Truth 位姿读取（17列）
│   │
│   ├── processing/
│   │   ├── timestamp_sync.py      # 时间戳对齐（O(N+M) 双指针线性插值）
│   │   ├── frame_processor.py     # 帧预处理（关键帧提取、resize、normalize）
│   │   └── label_builder.py       # GT → 动作标签（差分 + 四元数轴角转换）
│   │
│   ├── models/
│   │   ├── vision_encoder.py      # CLIP ViT-B/32 视觉 + 文本编码器封装
│   │   ├── vl_fusion.py           # Cross-Attention VL 融合（残差+LayerNorm+FFN）
│   │   └── action_predictor.py    # Transformer Encoder 动作预测器（ACT 简化版）
│   │
│   ├── quality/
│   │   ├── metrics.py             # 质量评估（滑动窗口 Nσ 异常值检测）
│   │   └── reporter.py            # HTML 报告 + Matplotlib 图表
│   │
│   └── utils/
│       ├── logger.py              # 结构化 JSON 日志（LoggerMixin）
│       └── config.py              # YAML 配置（点号属性访问）
│
└── scripts/
    └── download_data.sh           # EuRoC 数据集下载（MH_01_easy + MH_02_easy）
```

---

## 四、核心模块详解

### 4.1 数据摄取（Ingestion）

**设计思路：流式读取，借鉴 Flink Source 模式**

VideoReader 作为数据源，每次 yield 一帧，避免一次性加载全部 PNG 导致 OOM：

```python
class VideoReader:
    def stream_frames(self, max_frames=None) -> Iterator[VideoFrame]:
        """流式读取 PNG 序列，按 data.csv 时间戳顺序 yield"""
```

GtReader 加载 Ground Truth（17列 CSV）：

```
列定义：
  timestamp_ns           — 纳秒时间戳
  px, py, pz             — 全局坐标系位置（m）
  qw, qx, qy, qz         — 单位四元数姿态
  vx, vy, vz             — 全局坐标系速度（m/s）
  bwx, bwy, bwz          — 陀螺仪偏置（rad/s，校准用）
  bax, bay, baz          — 加速度计偏置（m/s²，校准用）

采样率：200Hz（与 IMU 相同）
精度：Leica 激光追踪仪，亚毫米级
```

---

### 4.2 时间戳对齐（核心算法）

**核心挑战**：视频 20fps（帧间隔 50ms）vs IMU 200Hz（采样间隔 5ms），采样率差 10×

**两种方法对比：**

| 方法 | 复杂度 | 误差上界 | 实测误差均值 |
|------|--------|----------|------------|
| 最近邻 | O(N log M) 二分查找 | 2.5ms | ~2.5ms |
| 线性插值 | O(N+M) 双指针 | 趋近 0 | **0.000ms** |

**线性插值原理（O(N+M) 双指针）：**

```
IMU 时间轴:  t₀ ─── t₁ ─── t₂ ─── t₃
视频帧时间:       v₁        v₂

对于视频帧 v₁（落在 [t₁, t₂] 之间）：
  w = (v₁ - t₁) / (t₂ - t₁)         ← v₁ 距 t₁ 的比例，[0,1]
  result = (1-w)·imu[t₁] + w·imu[t₂] ← 加权平均（折线插值）
  error  = min(v₁-t₁, t₂-v₁)         ← 到最近 IMU 点的距离

关键优化：视频帧单调递增，IMU 指针只需从上次位置向右推进
  → 总操作次数 N+M 而非 N×log(M)
```

---

### 4.3 Ground Truth 标签构造

**从原始位姿到训练标签的完整过程：**

#### 步骤一：插值对齐

GT（200Hz）→ 视频帧时间戳（20fps），使用向量化 searchsorted + 线性插值，O(N log M)

#### 步骤二：位置差分

```python
delta_pos[t] = pos[t+1] - pos[t]   # (3,) 单位 m

# 实测数值（50ms 帧间隔，MH_01_easy 慢速飞行）：
# dx ≈ 0.0001m, dy ≈ 0.0002m, dz ≈ 0.004m
```

#### 步骤三：旋转差分（四元数 → 轴角）

旋转不能直接相减，需要四元数相对旋转：

```python
# 1. 计算相对四元数
q_delta = q[t]⁻¹ ⊗ q[t+1]
#         ↑共轭      ↑四元数乘法（汉密顿积）

# 2. 取最短路径（qw >= 0）
if qw < 0: q_delta = -q_delta

# 3. 转换为轴角
angle  = 2 * arccos(qw)
axis   = [qx, qy, qz] / sin(angle/2)
result = axis * angle    # (3,) 单位 rad，模长 = 旋转角度
```

#### 步骤四：Action Chunking 窗口

```python
# 帧 i 的标签 = 未来 10 步的单步动作
label[i] = [delta[i], delta[i+1], ..., delta[i+9]]  # shape=(10, 7)

# 有效帧：0 到 N-11（最后 10 帧没有足够的未来步骤）
n_valid = N - horizon   # N=3682, horizon=10 → n_valid=3672
```

---

### 4.4 视觉编码器（CLIP ViT-B/32）

**为什么用 CLIP：**

1. 预训练权重（4亿图文对），无需训练，直接使用
2. 视觉和文本共享向量空间，天然支持 VL 融合
3. ViT 架构与具身智能主流方案（RT-2、Octo 等）对齐

**在训练中的角色：CLIP 完全冻结**

视觉特征在 `main.py` 中预计算并存为 npy 文件，训练时直接加载，无需重跑 CLIP 推理：

```
main.py 运行一次 → visual_feats.npy (N, 512)
train.py 直接加载 → 无 CLIP overhead
```

---

### 4.5 视觉-语言融合（VL Fusion）

**架构：Cross-Attention + 残差 + LayerNorm + FFN**

```
视觉特征 (B, 512) ── Q ─┐
                          ├→ Cross-Attention → +残差 → LayerNorm → FFN → +残差 → LayerNorm
语言特征 (512)    ── KV ─┘
                                                    ↓
                                           融合特征 (B, 512)
```

**语义**：视觉特征作为 Query，向语言特征"询问"与当前任务相关的信息，使视觉特征对任务描述敏感。

**训练时**：VLFusion 权重更新，让融合特征更有利于下游动作预测。

---

### 4.6 动作预测器（Action Predictor）

**架构**：Transformer Encoder，参考 ACT（Action Chunking with Transformers）简化版

```
输入 Token 序列（共 11 个 Token，每个 256 维）：
  Token 0:     VL 融合特征（512 → 256，Linear 投影）
  Token 1..10: IMU 历史窗口（最近 10 帧 × 6维 → 256，Linear 投影）
       ↓
  + 可学习位置编码 (1, 11, 256)
       ↓
  Transformer Encoder（4层，8头，dim=256，Pre-LN）
       ↓
  取 Token 0（经过 Self-Attention 已看过所有 IMU Token）
       ↓
  Linear(256, 7×10) → reshape → (B, 10, 7)
       ↓
  Sigmoid 约束 gripper 维度到 [0,1]
```

**Action Chunking 的优势**：

| 方式 | 推理模式 | 特点 |
|------|----------|------|
| 逐步预测 | 每步推理一次 Transformer | 累积误差大、抖动明显 |
| Action Chunking | 一次推理覆盖 10 步 | 更平滑、推理次数少 10× |

**7 维动作空间**：

```
action[0:3] = delta_position (dx, dy, dz)   末端位移（m）
action[3:6] = delta_rotation (rx, ry, rz)   旋转轴角（rad）
action[6]   = gripper_openness              夹爪开合 [0, 1]
```

---

### 4.7 训练流程

**策略：两段式，CLIP 冻结**

```
阶段一（main.py）：特征预计算
  CLIP ViT-B/32 → visual_feats.npy (N, 512)   ← 一次性，之后复用

阶段二（train.py）：参数训练
  可训练：VisionLanguageFusion + ActionPredictor
  冻结：  CLIP（直接加载预计算 npy，不重跑）

  每个 step：
    vis  = visual_feats[batch]          # 从 npy 加载
    imu  = imu_windows[batch]           # 滑动窗口
    fused = VLFusion(vis, text_feat)    # text_feat 固定
    pred  = ActionPredictor(fused, imu)
    loss  = MSELoss(pred, gt_labels)
    loss.backward()
    AdamW.step()
    CosineAnnealingLR.step()
```

**超参配置（configs/pipeline.yaml）**：

```yaml
training:
  epochs: 30
  batch_size: 64
  lr: 3e-4
  weight_decay: 1e-5
  val_split: 0.1
```

---

### 4.8 评估指标

| 指标 | 说明 | 单位 |
|------|------|------|
| `pos_mae_mm` | 位置差分 MAE，平均所有帧和预测步 | mm |
| `rot_mae_mrad` | 旋转差分 MAE，平均所有帧和预测步 | mrad |
| `per_step_mae` | 各预测步（1~10）的平均 MAE | mm/mrad 混合 |
| `per_dim_mae` | 每个动作维度的 MAE | mm 或 mrad |

**per_step_mae 的意义**：理想情况下，步骤 1 误差最小，随预测步增大误差递增（越远越难预测）。若误差曲线平坦，说明模型未有效学习时序依赖。

---

### 4.9 数据质量评估

| 指标 | 计算方法 | 阈值 |
|------|----------|------|
| 帧完整性 | actual / expected | > 99% |
| IMU 完整性 | actual / expected | > 99.5% |
| 同步误差均值 | mean(sync_errors) | < 5 ms |
| 同步误差标准差 | std(sync_errors) | < 3 ms |
| IMU 异常值率 | 滑动窗口 Nσ（见下） | < 0.1% |
| 视频模糊度 | 拉普拉斯方差 | > 100 为清晰 |

**IMU 异常值检测演进**：

```
全局 3σ（初版）：
  → 把飞行机动误判为异常（全局均值混合了所有飞行阶段）
  → 异常率 6.3%，远超阈值

滑动窗口 4.5σ（当前）：
  窗口大小 2s（400点），用局部统计量，只捕捉真实传感器毛刺
  实现：cumsum trick，O(N) 复杂度，无额外依赖
  → 异常率 0.07%，通过阈值 0.1%
```

---

## 五、数据方案

### 5.1 数据集选择：EuRoC MAV Dataset

| 特性 | 说明 |
|------|------|
| 数据来源 | ETH Zurich，机器人视觉领域标准基准数据集 |
| 传感器 | 立体相机（20fps）+ IMU（200Hz）+ Ground Truth（200Hz）|
| 下载方式 | 按序列独立下载，无需 ROS |
| 选定序列 | MH_01_easy（1.9GB）+ MH_02_easy（1.5GB）|
| 优势 | 真实传感器噪声、有 Ground Truth 可做训练标签 |

### 5.2 EuRoC 数据对 Pipeline 的映射

| 模块 | EuRoC 数据 | 说明 |
|------|-----------|------|
| VideoReader | `cam0/data.csv` + `cam0/data/` | 时间戳 + PNG 图像 |
| ImuReader | `imu0/data.csv` | 6维，纳秒时间戳 ÷ 1e6 转毫秒 |
| GtReader | `state_groundtruth_estimate0/data.csv` | 17列，位置+四元数+速度 |
| TimestampSync | 视频(20fps) vs IMU(200Hz) | 采样率差 10× |
| LabelBuilder | GT 位置+四元数差分 | 构造 7D 动作伪标签 |
| 语言指令 | 序列场景描述（人工配置） | `configs/pipeline.yaml → language` |

---

## 六、运行流程

### 完整三步闭环

```bash
# Step 1: 数据处理 + 特征预计算（约 45s，CPU）
python cmd/python/main.py

# Step 2: 模型训练（约 30 epochs × 几分钟/epoch，CPU）
python cmd/python/train.py

# Step 3A: 用训练权重做推理
python cmd/python/main.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt

# Step 3B: 对比推理结果与 Ground Truth
python cmd/python/evaluate.py --checkpoint data/output/checkpoints/MH_01_easy_best.pt
```

### 输出文件一览

```
data/output/
├── aligned/
│   ├── MH_01_easy_visual_feats.npy    (N, 512)  CLIP 视觉特征
│   ├── MH_01_easy_aligned_imu.npy     (N, 6)    对齐后 IMU
│   ├── MH_01_easy_fused_feats.npy     (N, 512)  VL 融合特征
│   ├── MH_01_easy_actions.npy         (N,10,7)  预测动作序列
│   └── MH_01_easy_video_ts.npy        (N,)      视频帧时间戳
├── checkpoints/
│   └── MH_01_easy_best.pt             验证集最优权重（含 epoch、val_loss、config）
└── reports/
    ├── report_MH_01_easy.html          数据质量报告
    ├── summary_MH_01_easy.json         质量指标 JSON
    ├── evaluation_MH_01_easy.json      动作预测评估指标 JSON
    └── img/
        ├── sync_error.png              时间戳同步误差分布
        ├── imu_accel.png / imu_gyro.png  IMU 时序图
        ├── blur.png                    视频模糊度曲线
        ├── eval_horizon_error.png      各预测步误差曲线
        └── eval_action_compare.png     GT vs 预测对比图
```

---

## 七、技术选型说明

| 选择 | 方案 | 理由 |
|------|------|------|
| 视觉编码器 | CLIP ViT-B/32（冻结） | 预训练、视觉-语言对齐、轻量、无需 GPU 也可用 |
| 动作预测 | 自实现 Transformer | 展示架构理解，比黑盒调用更有技术深度 |
| 训练策略 | 两段式（CLIP冻结 + VLFusion/Predictor可训练） | 降低计算开销，CLIP特征一次预计算复用 |
| 动作标签 | GT 差分（位置 + 四元数轴角） | EuRoC 自带高精度真值，无需额外标注 |
| 异常值检测 | 滑动窗口 Nσ（O(N) cumsum） | 适配飞行数据非平稳特性，无额外依赖 |
| 数据格式 | NumPy npy | 高性能存取，方便跨脚本共享特征 |
| 日志格式 | JSON 结构化日志 | 体现监控系统开发背景，可接 ELK |
| 配置管理 | YAML 点号访问 | 可扩展，工程实践标准 |
| 质量报告 | HTML + Matplotlib | 可直接打开，直观展示 |

**不用的技术及原因：**
- ❌ Flink/Spark：单机验证阶段无需分布式，保留接口作为扩展方向
- ❌ OpenVLA/Octo 直接调用：黑盒程度高，项目侧重理解数据流
- ❌ ROS：EuRoC 提供原始 CSV/PNG，无需 ROS 即可处理
- ❌ SLERP（球面线性插值）：GT 200Hz，帧间旋转变化极小（~1mrad），线性插值后归一化效果等同

---

## 八、技术讨论要点

**要点 1：数据工程能力向具身智能的迁移**

时序数据对齐、多路数据同步、高吞吐低延迟——这些是广告/监控数据平台的核心问题，也是机器人数据处理的核心问题。本项目通过 EuRoC 真实数据集，验证了这套数据工程能力在多模态机器人数据场景下的适用性。

**要点 2：时间戳对齐算法选择**

- 最近邻：误差最大 2.5ms，O(N log M)
- 线性插值：误差趋近 0，O(N+M) 双指针，利用单调性规避重复二分查找
- 实测：MH_01_easy 全 3682 帧，同步误差均值 **0.000ms**

**要点 3：IMU 异常值检测方法演进**

全局 3σ → 滑动窗口 4.5σ 的改进，体现了对"飞行数据非平稳"物理特性的深入理解：
飞行机动（转弯、加速）是正常信号而非异常，异常检测应基于局部统计而非全局统计。

**要点 4：训练数据来源与标签质量**

EuRoC Ground Truth 由 Leica 激光追踪仪提供，精度亚毫米级，远优于从 IMU 积分得到的位姿。差分标签（delta_pos + 轴角）符合机器人控制的增量指令范式，与 ACT、Diffusion Policy 等主流方案一致。

**要点 5：生产扩展路径**

- 数据摄取层 → Kafka Topic（VideoReader 变为 Consumer）
- 处理引擎 → Flink 流式处理（保留分层接口）
- 模型推理 → TorchScript / ONNX 导出 + GPU 加速
- 训练数据 → Open X-Embodiment（真实机械臂遥操作数据）替换飞行伪标签

---

## 九、注意事项

1. **运行顺序**：`main.py` → `train.py` → `evaluate.py`，每步依赖前一步的输出文件
2. **CLIP 首次下载**：约 300MB，本地缓存后离线可用；CPU 推理批量编码约 45s
3. **训练耗时**：CPU 上约 2-5min/epoch（取决于机器性能），30 epoch 共约 1-2h
4. **动作标签局限性**：当前 gripper 维度固定为 0（无人机无手爪），若迁移到真实机械臂任务需替换为遥操作数据集的真实开合标注
5. **评估指标参考值**：EuRoC MH_01_easy 慢速飞行，位置差分均值约 10mm/50ms，模型收敛后 MAE 预期在 10-20mm 量级

---

*文档版本：v2.0 | 2026-03*
