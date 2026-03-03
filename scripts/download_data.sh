#!/bin/bash
# 下载 EuRoC MAV Dataset 用于 embodied-data-flow 项目
# 数据集来源：ETH Zurich ASL
# 选定序列：MH_01_easy (1.9GB) + MH_02_easy (1.5GB)，共约 3.4GB

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw/euroc"
BASE_URL="http://rpg.ifi.uzh.ch/docs/IJRR17_Burri/datasets/EuRoC_MAV_dataset"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查依赖
check_deps() {
    for cmd in wget unzip; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "缺少依赖命令: $cmd"
            exit 1
        fi
    done
}

# 下载并解压单个序列
download_sequence() {
    local name="$1"   # 如 MH_01_easy
    local size="$2"   # 用于显示
    local zip_file="$DATA_DIR/${name}.zip"
    local target_dir="$DATA_DIR/${name}"

    if [ -d "$target_dir/mav0" ]; then
        log_warn "$name 已存在，跳过下载（删除 $target_dir 可重新下载）"
        return 0
    fi

    log_info "下载 $name ($size)..."
    wget --show-progress -q -O "$zip_file" "$BASE_URL/${name}.zip"

    log_info "解压 $name..."
    unzip -q "$zip_file" -d "$target_dir"

    # EuRoC zip 内层目录名为 mav0，解压后路径：target_dir/mav0
    # 如果 zip 内有额外嵌套目录，做修正
    local inner
    inner=$(find "$target_dir" -maxdepth 2 -name "mav0" -type d | head -1)
    if [ -n "$inner" ] && [ "$inner" != "$target_dir/mav0" ]; then
        mv "$inner" "$target_dir/mav0_tmp"
        # 移除多余的中间目录
        find "$target_dir" -mindepth 1 -maxdepth 1 -not -name "mav0_tmp" -exec rm -rf {} +
        mv "$target_dir/mav0_tmp" "$target_dir/mav0"
    fi

    rm -f "$zip_file"
    log_info "$name 下载完成 -> $target_dir"
}

# 验证序列完整性
verify_sequence() {
    local name="$1"
    local base="$DATA_DIR/$name/mav0"
    local ok=true

    for path in \
        "cam0/data.csv" \
        "cam0/data" \
        "imu0/data.csv" \
        "state_groundtruth_estimate0/data.csv"
    do
        if [ ! -e "$base/$path" ]; then
            log_warn "$name: 缺少 $path"
            ok=false
        fi
    done

    if $ok; then
        local img_count
        img_count=$(find "$base/cam0/data" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        log_info "$name 验证通过（cam0 图像数：$img_count）"
    else
        log_error "$name 数据不完整，请重新下载"
        return 1
    fi
}

main() {
    log_info "EuRoC MAV Dataset 下载脚本"
    log_info "目标目录：$DATA_DIR"
    echo ""

    check_deps
    mkdir -p "$DATA_DIR"

    download_sequence "MH_01_easy" "1.9GB"
    download_sequence "MH_02_easy" "1.5GB"

    echo ""
    log_info "验证数据完整性..."
    verify_sequence "MH_01_easy"
    verify_sequence "MH_02_easy"

    echo ""
    log_info "全部完成！数据目录结构："
    echo "  $DATA_DIR/"
    echo "  ├── MH_01_easy/mav0/{cam0, imu0, state_groundtruth_estimate0}"
    echo "  └── MH_02_easy/mav0/{cam0, imu0, state_groundtruth_estimate0}"
}

main "$@"
