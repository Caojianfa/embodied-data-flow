# Embodied Data Flow — 技术方案

> 具身智能多模态数据处理与动作预测平台

---

## 一、项目定位

### 1.1 背景与目标

| 维度 | 说明 |
|------|------|
| 项目定位 | **数据工程 + AI 并重**，用工程师视角构建完整数据闭环 |
| 核心目标 | 本地可运行、有实际输出、可清晰讲解每个模块 |

### 1.2 展示的关键能力

```
大数据工程能力（迁移自 Flink/Spark 经验）
  → 流式处理架构、高吞吐设计、时序数据对齐

多模态数据处理（新技能，项目中建立 credibility）
  → OpenCV 视频处理、IMU 信号处理、时间戳同步

AI 集成能力（基于 PyTorch/Transformer 经验）
  → 视觉特征提取（CLIP/ViT）
  → 视觉-语言模态融合
  → Transformer 动作预测

工程化能力（10 年数据工程背景）
  → 可观测性（结构化日志 + 性能指标）
  → 数据质量评估体系
  → 配置驱动的可扩展设计
```

---

## 二、整体架构

```
raw data (EuRoC MAV Dataset)
  ├── video (PNG 图像序列，20fps，含时间戳 CSV)
  ├── imu (CSV, 200Hz: acc_x/y/z, gyro_x/y/z，纳秒时间戳)
  └── language instruction (场景描述文本)
          │
          ▼
┌─────────────────────────────────────┐
│           Ingestion Layer           │
│   VideoReader  │  ImuReader         │
│   (流式，不一次加载) │  (Pandas 解析) │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│          Processing Engine          │
│                                     │
│  TimestampSync                      │
│  ├─ video: 20fps → 每帧时间戳        │
│  ├─ imu: 200Hz → 时间戳序列         │
│  └─ 线性插值对齐 → aligned_imu      │
│                                     │
│  FrameProcessor                     │
│  ├─ 抽关键帧（场景变化检测）          │
│  └─ 图像预处理（resize/normalize）   │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│             AI Models               │
│                                     │
│  VisionEncoder (CLIP ViT-B/32)      │
│  └─ 每帧 → 512 维视觉特征           │
│                                     │
│  VisionLanguageFusion               │
│  ├─ CLIPTextEncoder → 512 维        │
│  └─ Cross-Attention 融合 → 1024 维  │
│                                     │
│  ActionPredictor (Transformer)      │
│  ├─ input: 融合特征 + IMU 窗口      │
│  └─ output: 7 维动作序列 × T 步     │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│         Quality & Monitoring        │
│  ├─ 完整性检查（帧丢失率）           │
│  ├─ 同步质量（时间戳误差分布）       │
│  ├─ 视频质量（模糊度检测）           │
│  ├─ 吞吐量指标（帧/秒）             │
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
│   └── pipeline.yaml              # 全局配置（采样率、路径、模型参数等）
│
├── data/
│   ├── raw/                       # 原始数据（EuRoC MAV Dataset）
│   │   └── euroc/
│   │       ├── MH_01_easy/
│   │       │   └── mav0/
│   │       │       ├── cam0/data/          # 左目图像序列（PNG）
│   │       │       ├── cam0/data.csv       # 图像时间戳（ns）
│   │       │       ├── imu0/data.csv       # IMU 数据（200Hz，ns 时间戳）
│   │       │       └── state_groundtruth_estimate0/data.csv  # 地面真值位姿+速度
│   │       └── MH_02_easy/                # 结构同上
│   └── output/
│       ├── aligned/               # 对齐后的数据（npy 格式）
│       └── reports/               # HTML 质量报告 + 图表
│
├── docs/
│   └── tech_solution.md           # 本文档
│
├── cmd/
│   └── python/
│       └── main.py                # 命令行入口，串联整个 pipeline
│
├── pkg/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── video_reader.py        # 流式视频帧读取器
│   │   └── imu_reader.py          # CSV IMU 数据解析器
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── timestamp_sync.py      # 时间戳对齐算法（核心）
│   │   └── frame_processor.py     # 帧预处理（关键帧提取、图像处理）
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py      # CLIP 视觉编码器封装
│   │   ├── vl_fusion.py           # 视觉-语言融合模块
│   │   └── action_predictor.py    # Transformer 动作预测器
│   │
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── metrics.py             # 质量指标计算
│   │   └── reporter.py            # 报告生成（HTML + 图表）
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 结构化日志（JSON 格式）
│       └── config.py              # YAML 配置加载
│
└── scripts/
    └── download_data.sh           # 下载 EuRoC MAV 数据集（MH_01_easy + MH_02_easy）
```

---

## 四、核心模块详解

### 4.1 数据摄取（Ingestion）

**设计思路：流式读取，不一次性加载**

这里借鉴 Flink/Kafka 的生产者-消费者模式，VideoReader 作为数据源，每次 yield 一帧，避免 OOM。

```python
# pkg/ingestion/video_reader.py 核心接口
# EuRoC 格式：读取 cam0/data.csv 获取时间戳列表，逐帧加载对应 PNG 文件
class VideoReader:
    def stream_frames(self) -> Iterator[VideoFrame]:
        """流式读取 PNG 图像序列，每次 yield 一帧（按 data.csv 时间戳顺序）"""
        ...

# VideoFrame 数据结构
@dataclass
class VideoFrame:
    timestamp_ms: float     # 时间戳（毫秒，从 EuRoC 纳秒转换）
    frame_id: int           # 帧序号
    image: np.ndarray       # BGR 图像 (480, 752, 3)  EuRoC 分辨率
    fps: float              # 原始帧率（EuRoC cam0 = 20fps）
```

**IMU 数据格式（EuRoC imu0/data.csv）：**

```
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
1403636579758555392,-0.099134701513277898,0.14730578886071584,...
1403636579763555392,-0.099134701513277898,0.14730578886071584,...
# 时间戳为纳秒，ImuReader 需转换为毫秒：timestamp_ms = timestamp_ns / 1e6
```

---

### 4.2 时间戳对齐（核心算法）

**核心挑战：**

- 视频（EuRoC cam0）：20 FPS → 时间间隔 50ms/帧
- IMU（EuRoC imu0）：200 Hz → 时间间隔 5ms/样本
- 采样率差 10 倍，需要精确对齐（比常规 30fps vs 200Hz 场景挑战更大）

**算法：线性插值（O(N+M) 双指针）**

```python
def align_modalities(video_ts, imu_ts, imu_data, method='linear'):
    """
    将 IMU 数据对齐到视频帧时间戳

    Args:
        video_ts: 视频帧时间戳 shape=(N_frames,)
        imu_ts:   IMU 时间戳  shape=(N_imu,)
        imu_data: IMU 数据    shape=(N_imu, 6)
        method:   'nearest' | 'linear'

    Returns:
        aligned_imu:  shape=(N_frames, 6)  对齐后的 IMU 数据
        sync_errors:  shape=(N_frames,)    每帧的同步误差（ms）
    """
```

**线性插值原理：**

```
IMU 时间轴:  t0 ---- t1 ---- t2 ---- t3
视频帧时间:       v1        v2

对于视频帧 v1：找到满足 t1 < v1 < t2 的 IMU 采样对
插值权重 w = (v1 - t1) / (t2 - t1)
插值结果 = (1-w) * imu[t1] + w * imu[t2]
同步误差 = min(v1-t1, t2-v1)
```

**质量评估：**
- 期望同步误差：均值 < 5ms（IMU 5ms 间隔的一半）
- 如果均值 > 10ms，说明时间戳对齐有问题

---

### 4.3 视觉编码器（CLIP ViT-B/32）

**为什么用 CLIP？**

1. 预训练模型，无需训练
2. 视觉和文本在同一向量空间 → 天然支持视觉-语言融合
3. ViT 架构与具身智能主流方案（RT-2, Octo 等）对齐

```python
# pkg/models/vision_encoder.py
class VisionEncoder:
    def __init__(self, model_name='ViT-B/32', device='cpu'):
        self.model, self.preprocess = clip.load(model_name, device)

    def encode_frame(self, image: np.ndarray) -> torch.Tensor:
        """单帧编码，返回 512 维特征向量"""
        ...

    def encode_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """批量编码，返回 (B, 512) 特征矩阵"""
        ...
```

**输出示例：**
- 输入：视频帧 (480, 640, 3)
- 输出：特征向量 (512,)

---

### 4.4 视觉-语言融合（VL Fusion）

**设计：Cross-Attention 融合**

```
视觉特征 (512) ──┐
                  ├→ Cross-Attention → LayerNorm → FFN → 融合特征 (512)
语言特征 (512) ──┘
```

```python
# pkg/models/vl_fusion.py
class VisionLanguageFusion(nn.Module):
    """
    视觉-语言多模态融合模块

    视觉特征作为 Query，语言特征作为 Key/Value
    语义上：用语言指令"引导"视觉特征关注相关区域
    """
    def __init__(self, dim=512, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
```

**使用示例：**

```python
fusion = VisionLanguageFusion(dim=512)

# 语言：EuRoC 场景描述（由序列名和场景生成，如 Machine Hall 序列）
text = "Flying over Machine Hall, approach the corridor"

# 融合
visual_feat = vision_encoder.encode_frame(frame)   # (512,)
text_feat   = clip_text_encode(text)               # (512,)
fused       = fusion(visual_feat, text_feat)       # (512,)
```

---

### 4.5 动作预测模型（Action Predictor）

**架构：Transformer Encoder-Decoder**

参考 ACT（Action Chunking with Transformers）的思路，但大幅简化：

```
输入序列：
  [视觉-语言融合特征]  → Token 0（场景理解）
  [IMU 窗口数据 ×10]  → Token 1..10（运动状态，过去 0.05s）
  [机器人当前状态]     → Token 11（关节位置）

Transformer Encoder (4层, 8头, dim=256)

动作解码头：
  → Linear(256, 7×T)
  → reshape → (T, 7)    # T=10步预测，7维动作（6DoF + gripper）
```

**7 维动作空间：**

```
action[0:3]  = delta_position (x, y, z)      末端执行器位移
action[3:6]  = delta_rotation (rx, ry, rz)   末端执行器旋转
action[6]    = gripper_openness              夹爪开合 [0, 1]
```

**Action Chunking（动作分块预测）：**

> 一次性预测未来 T 步动作，而非每步预测一次
>
> 优点：
> - 减少 Transformer 推理次数
> - 预测更平滑，避免抖动
> - 与 ACT、RT-1 等主流方案一致

---

### 4.6 数据质量评估

| 指标 | 计算方法 | 期望值 |
|------|----------|--------|
| **帧完整性** | actual_frames / expected_frames | > 99% |
| **IMU 完整性** | actual_samples / expected_samples | > 99.5% |
| **同步误差均值** | mean(sync_errors) | < 5 ms |
| **同步误差方差** | std(sync_errors) | < 3 ms |
| **视频模糊度** | 拉普拉斯方差 (Laplacian variance) | > 100 为清晰 |
| **IMU 异常值率** | 超过 3σ 的采样点比例 | < 0.1% |
| **处理吞吐量** | frames / elapsed_seconds | 目标 > 100 FPS |

**模糊度检测原理（拉普拉斯方差）：**

```python
def detect_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # variance 越小 → 越模糊
    # < 50: 严重模糊，< 100: 轻微模糊，> 100: 清晰
    return variance
```

---

## 五、数据方案

### 5.1 数据集选择：EuRoC MAV Dataset

使用 ETH Zurich 发布的 **EuRoC MAV Dataset** 作为真实数据集：

| 特性 | 说明 |
|------|------|
| 数据来源 | ETH Zurich，机器人视觉领域标准基准数据集 |
| 传感器 | 立体相机（20fps）+ IMU（200Hz）+ 地面真值位姿 |
| 下载方式 | 按序列独立下载（11 个序列可单独获取） |
| 选定序列 | MH_01_easy（1.9GB）+ MH_02_easy（1.5GB），共 3.4GB |
| 格式 | CSV 时间戳文件 + PNG 图像序列，无需 ROS |
| 优势 | 真实传感器噪声、真实时间戳抖动、有地面真值可做伪标签 |

### 5.2 下载方式

运行 `scripts/download_data.sh`：

```bash
#!/bin/bash
mkdir -p data/raw/euroc

# EuRoC MH_01_easy (1.9GB)
wget -P data/raw/euroc \
  http://rpg.ifi.uzh.ch/docs/IJRR17_Burri/datasets/EuRoC_MAV_dataset/MH_01_easy.zip
unzip data/raw/euroc/MH_01_easy.zip -d data/raw/euroc/MH_01_easy

# EuRoC MH_02_easy (1.5GB)
wget -P data/raw/euroc \
  http://rpg.ifi.uzh.ch/docs/IJRR17_Burri/datasets/EuRoC_MAV_dataset/MH_02_easy.zip
unzip data/raw/euroc/MH_02_easy.zip -d data/raw/euroc/MH_02_easy
```

> 可在后台挂起下载：`nohup bash scripts/download_data.sh &`

### 5.3 EuRoC 数据对 Pipeline 的影响

| 模块 | 适配说明 |
|------|----------|
| VideoReader | 读取 `cam0/data.csv` 获取时间戳列表，逐帧加载 PNG |
| ImuReader | 读取 `imu0/data.csv`，时间戳从纳秒转毫秒（÷1e6） |
| TimestampSync | 20fps vs 200Hz（采样率差 10×，真实时间戳抖动） |
| VL Fusion | 使用序列名/场景描述生成语言输入（无需人工标注） |
| ActionPredictor | 使用 `state_groundtruth_estimate0` 中的速度作为伪动作标签（6D）|

---

## 六、实施计划

| 阶段 | 预估时长 | 内容 | 完成标准 | 前置条件 |
|------|----------|------|----------|----------|
| 0. 基础结构 | 0.5h | 目录、依赖、配置文件 | `pip install` 通过 | - |
| 1. 下载数据 | 后台下载 | `download_data.sh`（MH_01+MH_02，共 3.4GB） | 数据目录结构完整 | 宽带环境 |
| 2. 数据摄取 | 1.5h | VideoReader + ImuReader | 能流式读取 EuRoC 数据 | 数据下载完成 |
| 3. 时间戳对齐 | 1.5h | TimestampSync 算法 | 对齐误差 < 5ms | - |
| 4. 帧处理 | 0.5h | FrameProcessor | 输出预处理后的帧 | - |
| 5. 视觉编码器 | 1h | CLIP 封装 | 输出 512 维特征 | - |
| 6. VL 融合 | 1h | Cross-Attention 模块 | 能融合视觉+语言 | - |
| 7. 动作预测 | 1.5h | Transformer 预测器 | 能输出动作序列 | - |
| 8. 质量评估 | 1h | Metrics + Reporter | 生成 HTML 报告 | - |
| 9. Pipeline 整合 | 1h | main.py 串联所有模块 | 端到端一条命令运行 | - |
| 10. 文档 + README | 0.5h | 使用说明 | 他人可快速上手 | - |
| **合计** | **约 11h** | - | - | 数据可提前挂起下载 |

---

## 七、技术选型说明

| 选择 | 方案 | 理由 |
|------|------|------|
| 视觉编码器 | CLIP ViT-B/32 | 预训练、视觉-语言对齐、轻量 |
| 动作预测 | 自实现 Transformer | 展示架构理解，比直接用 ACT 更有技术深度 |
| 数据格式 | NumPy npy | 高性能存取，格式简单易解释 |
| 日志格式 | JSON 结构化日志 | 体现监控系统开发背景 |
| 配置管理 | YAML | 可扩展，工程实践 |
| 质量报告 | HTML + Matplotlib | 可直接打开，直观展示 |

**不用的技术及原因：**
- ❌ Flink/Spark：本项目侧重单机 Pipeline 验证，流式框架可作为后续扩展方向
- ❌ OpenVLA/Octo 直接加载：黑盒程度高，不利于理解数据流；项目侧重数据工程而非模型复现
- ❌ ROS：环境配置复杂，EuRoC 提供原始 CSV/PNG 格式，无需 ROS 即可处理
- ❌ 分布式：范围过大，保留设计文档说明扩展方向即可

---

## 八、技术讨论要点

**要点 1：数据工程能力向具身智能的迁移**

时序数据对齐、多路数据同步、高吞吐低延迟——这些是广告/监控数据平台的核心问题，也是机器人数据处理的核心问题。本项目通过 EuRoC 真实数据集，验证了这套数据工程能力在多模态机器人数据场景下的适用性。

**要点 2：时间戳对齐算法选择**

- **最近邻对齐**：误差最大 2.5ms（IMU 5ms 间隔的一半），实现简单
- **线性插值对齐**：误差降至亚毫秒级，对运动状态变化频繁的场景更准确
- EuRoC 的 20fps vs 200Hz（10× 差异）比常见的 30fps vs 200Hz 场景更有挑战性，可通过质量报告中的误差分布直方图量化对比

**要点 3：动作预测模型定位**

本项目的 ActionPredictor 是架构验证模型，而非 SOTA 策略模型：
- RT-1 使用 token 化动作空间
- ACT 使用 CVAE + Transformer
- 本项目使用简化 Transformer Encoder，配合 EuRoC 地面真值速度作为伪标签，目标是 end-to-end 验证数据 Pipeline 的正确性

**要点 4：生产扩展路径**

- 数据摄取层 → Kafka Topic（VideoReader 变为 Consumer）
- 处理引擎 → Flink 流式处理（保留分层架构接口）
- 模型推理 → 批处理 + GPU 加速
- 当前架构的分层设计即为此扩展预留接口

---

## 九、预期输出展示

**终端运行输出：**

```
[YYYY-MM-DD INFO] pipeline started config=pipeline.yaml
[YYYY-MM-DD INFO] video loaded frames=600 fps=20 duration=30s  # EuRoC cam0 20fps
[YYYY-MM-DD INFO] imu loaded samples=6000 rate=200Hz
[YYYY-MM-DD INFO] timestamp alignment completed
  sync_error_mean=2.3ms sync_error_std=1.1ms frame_drop_rate=0.2%
[YYYY-MM-DD INFO] vision encoding completed throughput=145 frames/sec
[YYYY-MM-DD INFO] vl fusion completed
[YYYY-MM-DD INFO] action prediction completed actions_shape=(600, 10, 7)
[YYYY-MM-DD INFO] quality report generated path=data/output/reports/report.html
[YYYY-MM-DD INFO] pipeline finished elapsed=6.2s
```

**质量报告内容（HTML）：**

1. 数据概览：帧数、IMU采样数、时长
2. 时间戳对齐误差分布（直方图）
3. IMU 各轴数据时序图（检测异常）
4. 视频模糊度逐帧曲线（找到模糊帧）
5. 吞吐量随时间变化曲线（性能监控）

---

## 十、注意事项 & 风险

1. **CLIP 模型首次下载需要网络**（约 300MB），本地缓存后离线可用
2. **无 GPU 也可运行**，CLIP ViT-B/32 在 CPU 上推理单帧约 100-200ms，批量处理 30s 视频约 2-3 分钟
3. **动作预测模型使用 EuRoC 地面真值速度作为伪动作标签**，需要注意这是 Pipeline 架构验证，而非训练完整的策略模型；如需真实策略模型，需引入更丰富的遥操作数据集
4. 如果 CLIP 依赖有问题，视觉编码器可以退回 `torchvision` 的 ResNet-18（无需特殊安装）

---

*文档版本：v1.1 | 2026-03*
