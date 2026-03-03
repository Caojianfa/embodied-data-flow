"""
生成 EuRoC 格式的合成测试数据

输出目录结构与真实 EuRoC 完全一致，可直接用于 Pipeline 测试：

  data/raw/euroc/MH_01_easy/
  └── mav0/
      ├── cam0/
      │   ├── data.csv           # 时间戳（纳秒）
      │   └── data/              # PNG 图像序列
      ├── imu0/
      │   └── data.csv           # IMU 数据（200Hz）
      └── state_groundtruth_estimate0/
          └── data.csv           # 地面真值位姿+速度

合成数据特征：
  - 相机：20fps，752×480，模拟飞行轨迹（渐变背景+随机纹理+真实噪声）
  - IMU：200Hz，重力 + 正弦运动 + 高斯噪声 + 零偏漂移
  - 时间戳：真实抖动（±0.3ms），纳秒精度

用法：
  python scripts/generate_euroc_sample.py                    # 默认 30s，存到 MH_01_easy
  python scripts/generate_euroc_sample.py --duration 10      # 只生成 10s
  python scripts/generate_euroc_sample.py --sequence TEST_01 # 自定义序列名
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 参数 ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 EuRoC 格式合成数据")
    p.add_argument("--duration", type=float, default=30.0, help="序列时长（秒），默认 30")
    p.add_argument("--sequence", default="MH_01_easy", help="序列名称，默认 MH_01_easy")
    p.add_argument("--fps", type=float, default=20.0, help="相机帧率，默认 20")
    p.add_argument("--imu-rate", type=float, default=200.0, help="IMU 采样率，默认 200")
    p.add_argument("--width", type=int, default=752)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── 图像生成 ──────────────────────────────────────────────────────

def make_frame(
    t: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成单帧图像，模拟机器人从走廊飞行的视角：
    - 渐变背景色（随时间缓慢变化）
    - 随机纹理斑块（模拟墙面特征点）
    - 10× 帧丢弃噪声（偶发）
    - 均匀高斯噪声
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景：根据时间变化的渐变
    bg_r = int(40 + 30 * np.sin(t * 0.3))
    bg_g = int(50 + 20 * np.cos(t * 0.2))
    bg_b = int(60 + 25 * np.sin(t * 0.15 + 1.0))
    img[:] = (bg_b, bg_g, bg_r)

    # 水平线（模拟地平线）
    horizon_y = int(height * 0.55 + 30 * np.sin(t * 0.5))
    img[horizon_y:, :] = (bg_b // 3, bg_g // 3, bg_r // 3)

    # 特征点斑块（固定种子，保证跨帧一致）
    rng_tex = np.random.default_rng(42)
    n_features = 60
    fxs = rng_tex.integers(20, width - 20, n_features)
    fys = rng_tex.integers(20, height - 20, n_features)
    fcs = rng_tex.integers(80, 220, (n_features, 3))

    # 根据时间给特征点加轻微位移（模拟视差）
    dx = int(15 * np.sin(t * 0.4))
    dy = int(8 * np.cos(t * 0.3))
    for fx, fy, fc in zip(fxs, fys, fcs):
        cx = int(np.clip(fx + dx, 5, width - 6))
        cy = int(np.clip(fy + dy, 5, height - 6))
        r = rng_tex.integers(3, 8)
        cv2.circle(img, (cx, cy), r, fc.tolist(), -1)

    # 走廊透视线（模拟机器视角）
    vp_x = width // 2 + int(20 * np.sin(t * 0.25))
    vp_y = int(height * 0.45)
    for ang in [-60, -30, 0, 30, 60]:
        ex = vp_x + int(width * 0.6 * np.cos(np.radians(ang + 90)))
        ey = 0 if ang % 30 == 0 else height
        cv2.line(img, (vp_x, vp_y), (ex, ey), (100, 100, 100), 1, cv2.LINE_AA)

    # 偶发模糊帧（约 2% 概率）
    if rng.random() < 0.02:
        img = cv2.GaussianBlur(img, (15, 15), 5)

    # 均匀高斯噪声
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ── IMU 生成 ──────────────────────────────────────────────────────

def make_imu(
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成 IMU 数据，shape=(N, 7)，列：[ts_ns, wx, wy, wz, ax, ay, az]

    模拟特征：
    - 加速度：重力(0,0,9.81) + 正弦运动加速度 + 高斯噪声
    - 角速度：正弦旋转 + 零偏漂移（随时间缓慢变化）+ 高斯噪声
    """
    t = timestamps_ns / 1e9  # 转为秒
    n = len(t)

    # 角速度（rad/s）
    wx = 0.05 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.005, n)
    wy = 0.03 * np.cos(2 * np.pi * 0.2 * t) + rng.normal(0, 0.005, n)
    wz = 0.02 * np.sin(2 * np.pi * 0.15 * t + 0.5) + rng.normal(0, 0.003, n)

    # 零偏漂移（缓慢变化）
    drift = 0.001 * t
    wx += drift
    wy += 0.5 * drift

    # 加速度（m/s²）
    ax = 0.3 * np.sin(2 * np.pi * 0.4 * t) + rng.normal(0, 0.02, n)
    ay = 0.2 * np.cos(2 * np.pi * 0.3 * t) + rng.normal(0, 0.02, n)
    az = 9.81 + 0.1 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.02, n)

    return np.stack([timestamps_ns, wx, wy, wz, ax, ay, az], axis=1)


# ── 地面真值生成 ──────────────────────────────────────────────────

def make_groundtruth(timestamps_ns: np.ndarray) -> np.ndarray:
    """
    生成地面真值，shape=(N, 17)，EuRoC groundtruth 格式：
    [ts_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
    """
    t = timestamps_ns / 1e9
    n = len(t)

    # 位置（简单圆形轨迹）
    px = 1.5 * np.sin(0.2 * t)
    py = 1.5 * np.cos(0.2 * t) - 1.5
    pz = 0.5 * t  # 缓慢上升

    # 姿态（单位四元数，简化为接近恒定）
    qw = np.ones(n) * 0.999
    qx = 0.01 * np.sin(0.1 * t)
    qy = 0.01 * np.cos(0.1 * t)
    qz = 0.005 * np.sin(0.05 * t)

    # 速度
    vx = 0.3 * np.cos(0.2 * t)
    vy = -0.3 * np.sin(0.2 * t)
    vz = np.ones(n) * 0.5

    # 零偏（小值）
    zeros = np.zeros(n)
    bwx, bwy, bwz = zeros + 0.001, zeros + 0.001, zeros
    bax, bay, baz = zeros, zeros, zeros

    return np.stack(
        [timestamps_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz,
         bwx, bwy, bwz, bax, bay, baz],
        axis=1,
    )


# ── 主函数 ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 构建目录
    base = ROOT / "data" / "raw" / "euroc" / args.sequence / "mav0"
    cam_dir = base / "cam0" / "data"
    cam_csv = base / "cam0" / "data.csv"
    imu_csv = base / "imu0" / "data.csv"
    gt_csv = base / "state_groundtruth_estimate0" / "data.csv"

    for d in [cam_dir, base / "imu0", base / "state_groundtruth_estimate0"]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 相机时间戳（20fps，含真实抖动 ±0.3ms）
    n_frames = int(args.duration * args.fps)
    interval_ns = int(1e9 / args.fps)
    t0_ns = 1_403_636_579_758_555_392  # EuRoC 真实起始时间戳

    cam_ts_ns = np.array([
        t0_ns + i * interval_ns + rng.integers(-300_000, 300_001)
        for i in range(n_frames)
    ], dtype=np.int64)

    # ── IMU 时间戳（200Hz，含抖动 ±0.1ms）
    imu_interval_ns = int(1e9 / args.imu_rate)
    n_imu = int(args.duration * args.imu_rate) + 10  # 多几个保证覆盖
    imu_ts_ns = np.array([
        t0_ns + i * imu_interval_ns + rng.integers(-100_000, 100_001)
        for i in range(n_imu)
    ], dtype=np.int64)

    # ── 生成图像
    print(f"[INFO] 生成 {n_frames} 帧图像（{args.fps}fps，{args.width}×{args.height}）...")
    cam_rows = []
    for i, ts in enumerate(cam_ts_ns):
        filename = f"{ts}.png"
        img = make_frame(ts / 1e9, args.width, args.height, rng)
        cv2.imwrite(str(cam_dir / filename), img)
        cam_rows.append(f"{ts},{filename}")
        if (i + 1) % 100 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames} 帧完成")

    # cam0/data.csv
    with open(cam_csv, "w") as f:
        f.write("#timestamp [ns],filename\n")
        f.write("\n".join(cam_rows) + "\n")
    print(f"[INFO] cam0/data.csv 已写入 ({n_frames} 行)")

    # ── 生成 IMU
    print(f"[INFO] 生成 {n_imu} 个 IMU 采样（{args.imu_rate}Hz）...")
    imu_data = make_imu(imu_ts_ns, rng)
    with open(imu_csv, "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                "a_RS_S_z [m s^-2]\n")
        for row in imu_data:
            ts = int(row[0])
            vals = ",".join(f"{v:.10f}" for v in row[1:])
            f.write(f"{ts},{vals}\n")
    print(f"[INFO] imu0/data.csv 已写入 ({n_imu} 行)")

    # ── 生成地面真值
    print("[INFO] 生成地面真值...")
    gt_ts_ns = imu_ts_ns  # groundtruth 与 IMU 同频
    gt_data = make_groundtruth(gt_ts_ns)
    with open(gt_csv, "w") as f:
        f.write("#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
                "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
                "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],"
                "b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],"
                "b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n")
        for row in gt_data:
            ts = int(row[0])
            vals = ",".join(f"{v:.10f}" for v in row[1:])
            f.write(f"{ts},{vals}\n")
    print(f"[INFO] state_groundtruth_estimate0/data.csv 已写入")

    print(f"\n[DONE] 合成数据已生成：")
    print(f"  序列名：{args.sequence}")
    print(f"  时长：{args.duration}s")
    print(f"  图像帧数：{n_frames}（{args.fps}fps）")
    print(f"  IMU 采样：{n_imu}（{args.imu_rate}Hz）")
    print(f"  路径：{base.parent}")
    print(f"\n运行 Pipeline：")
    print(f"  python cmd/python/main.py --sequence {args.sequence}")


if __name__ == "__main__":
    main()
