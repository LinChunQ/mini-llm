#!/usr/bin/env bash
set -euo pipefail

# 始终切到仓库根目录，确保从任意路径启动都能找到项目文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TMPDIR="${TMPDIR:-$SCRIPT_DIR/.tmp}"
mkdir -p "$TMPDIR"
export TMPDIR

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv-linux}"
DATA_FILE="${DATA_FILE:-dataset/poetry_train.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-auto}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
AMP_MODE="${AMP_MODE:-auto}"
GPU_SELECTION_MODE="${GPU_SELECTION_MODE:-auto}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_TRUSTED_HOST="${TORCH_TRUSTED_HOST:-}"

# 检查目标分区剩余空间，避免依赖下载到一半才因为磁盘不足失败。
check_free_space() {
    local target_path="$1"
    local target_name="$2"
    local free_kb

    free_kb="$(df -Pk "$target_path" | awk 'NR==2 {print $4}')"
    if [ -z "$free_kb" ]; then
        echo "无法检查磁盘空间：$target_name"
        exit 1
    fi

    if [ "$free_kb" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
        local free_gb
        free_gb="$(awk -v kb="$free_kb" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
        echo "磁盘空间不足：$target_name 仅剩 ${free_gb}G，至少需要 ${MIN_FREE_GB}G"
        exit 1
    fi
}

# 自动选择空闲显存最多的 GPU，避免默认落到被其他服务占满的 0 号卡。
select_training_gpu() {
    local best_line
    local best_index
    local best_free
    local best_used

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        return
    fi

    case "$GPU_SELECTION_MODE" in
        auto)
            ;;
        off)
            echo "Skip GPU auto-selection because GPU_SELECTION_MODE=off"
            return
            ;;
        *)
            echo "不支持的 GPU_SELECTION_MODE：$GPU_SELECTION_MODE，可选值为 auto / off"
            exit 1
            ;;
    esac

    best_line="$(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | sort -t',' -k2,2nr | head -n 1)"
    if [ -z "$best_line" ]; then
        return
    fi

    best_index="$(printf '%s' "$best_line" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
    best_free="$(printf '%s' "$best_line" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
    best_used="$(printf '%s' "$best_line" | awk -F',' '{gsub(/ /, "", $3); print $3}')"

    if [ -n "$best_index" ]; then
        export CUDA_VISIBLE_DEVICES="$best_index"
        echo "Auto-selected GPU index=$best_index free_mem=${best_free}MiB used_mem=${best_used}MiB"
    fi
}

# 单独解析 torch 下载源，避免 Linux 上默认安装 PyPI 的 CUDA 依赖集合。
resolve_torch_index_url() {
    if [ -n "$TORCH_INDEX_URL" ]; then
        printf '%s\n' "$TORCH_INDEX_URL"
        return
    fi

    case "$TORCH_CHANNEL" in
        cpu)
            printf '%s\n' "https://download.pytorch.org/whl/cpu"
            ;;
        cu118|cu121|cu124)
            printf '%s\n' "https://download.pytorch.org/whl/$TORCH_CHANNEL"
            ;;
        pypi)
            printf '%s\n' "$PIP_INDEX_URL"
            ;;
        skip)
            printf '%s\n' ""
            ;;
        *)
            echo "不支持的 TORCH_CHANNEL：$TORCH_CHANNEL，可选值为 cpu / cu118 / cu121 / cu124 / pypi / skip"
            exit 1
            ;;
    esac
}

# torch 官方索引和通用 PyPI 镜像的 trusted host 需要分别处理。
resolve_torch_trusted_host() {
    if [ -n "$TORCH_TRUSTED_HOST" ]; then
        printf '%s\n' "$TORCH_TRUSTED_HOST"
        return
    fi

    case "$TORCH_CHANNEL" in
        cpu|cu118|cu121|cu124)
            printf '%s\n' "download.pytorch.org"
            ;;
        pypi)
            printf '%s\n' "$PIP_TRUSTED_HOST"
            ;;
        skip)
            printf '%s\n' ""
            ;;
        *)
            echo "不支持的 TORCH_CHANNEL：$TORCH_CHANNEL"
            exit 1
            ;;
    esac
}

# torch 默认单独安装为 CPU 版本；需要 GPU 时由环境变量显式切换。
install_torch_package() {
    local torch_index_url
    local torch_trusted_host

    if [ "$TORCH_CHANNEL" = "skip" ]; then
        echo "Skip torch install because TORCH_CHANNEL=skip"
        return
    fi

    torch_index_url="$(resolve_torch_index_url)"
    torch_trusted_host="$(resolve_torch_trusted_host)"

    echo "Installing torch==$TORCH_VERSION via channel: $TORCH_CHANNEL"
    "$PYTHON_CMD" -m pip install "torch==$TORCH_VERSION" \
        --timeout "$PIP_DEFAULT_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        --no-cache-dir \
        --index-url "$torch_index_url" \
        --trusted-host "$torch_trusted_host"
}

# 非 torch 依赖继续复用 requirements.txt，但过滤掉 torch 避免重复安装。
install_non_torch_requirements() {
    local filtered_requirements="$TMPDIR/requirements.no-torch.txt"

    awk '
        BEGIN { IGNORECASE = 1 }
        {
            line = $0
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line == "" || line ~ /^#/) next
            if (line ~ /^torch([<>=!~].*)?$/) next
            print $0
        }
    ' requirements.txt > "$filtered_requirements"

    if [ -s "$filtered_requirements" ]; then
        "$PYTHON_CMD" -m pip install -r "$filtered_requirements" \
            --timeout "$PIP_DEFAULT_TIMEOUT" \
            --retries "$PIP_RETRIES" \
            --no-cache-dir \
            --index-url "$PIP_INDEX_URL" \
            --trusted-host "$PIP_TRUSTED_HOST"
    fi
}

# 检查 train.py 是否支持某个命令行参数，兼容脚本与训练代码未同时更新的场景。
train_supports_arg() {
    local arg_name="$1"
    "$PYTHON_CMD" train.py --help 2>/dev/null | grep -q -- "$arg_name"
}

PYTHON_CMD="$PYTHON_BIN"

# 优先复用当前 conda 环境；如果没有可用环境，则在本地创建 Linux 虚拟环境。
if [ "$BOOTSTRAP_ENV" != "venv" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    PYTHON_CMD="$VENV_DIR/bin/python"
fi

# 打印实际使用的解释器，便于排查环境串用问题。
echo "Using Python: $PYTHON_CMD"

# 先检查项目目录和临时目录所在分区空间，提前暴露磁盘不足问题。
check_free_space "$SCRIPT_DIR" "项目目录"
check_free_space "$TMPDIR" "临时目录"

# CUDA 分配器开启可扩展段，能降低长时间训练后的显存碎片问题。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
select_training_gpu

need_torch=0
need_runtime_deps=0

# 分开检测依赖缺失项，避免 matplotlib/tqdm 缺失时重复下载 torch。
if ! "$PYTHON_CMD" -c "import torch" >/dev/null 2>&1; then
    need_torch=1
fi

# 训练运行所需的小依赖单独检查，便于定位安装失败原因。
if ! "$PYTHON_CMD" -c "import matplotlib, tqdm" >/dev/null 2>&1; then
    need_runtime_deps=1
fi

if [ "$need_torch" -eq 1 ] || [ "$need_runtime_deps" -eq 1 ]; then
    "$PYTHON_CMD" -m pip install --upgrade pip \
        --timeout "$PIP_DEFAULT_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        --trusted-host "$PIP_TRUSTED_HOST"
fi

if [ "$need_torch" -eq 1 ]; then
    if [ "$TORCH_CHANNEL" = "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
        echo "Detected NVIDIA GPU, but TORCH_CHANNEL is cpu. Set TORCH_CHANNEL=cu121 (or cu118/cu124) if you want CUDA training."
    fi
    install_torch_package
fi

if [ "$need_runtime_deps" -eq 1 ]; then
    install_non_torch_requirements
fi

train_batch_size="$BATCH_SIZE"
if [ -z "$train_batch_size" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # GPU 默认使用更保守的单步 batch，再用梯度累积维持有效 batch。
        train_batch_size=4
    else
        train_batch_size=16
    fi
fi

amp_args=()
case "$AMP_MODE" in
    auto)
        ;;
    on)
        amp_args+=(--amp)
        ;;
    off)
        amp_args+=(--no-amp)
        ;;
    *)
        echo "不支持的 AMP_MODE：$AMP_MODE，可选值为 auto / on / off"
        exit 1
        ;;
esac

train_extra_args=()

# 仅在 train.py 已支持新参数时再透传，避免服务器上脚本和代码版本不一致时报错。
if train_supports_arg "--grad-accum-steps"; then
    train_extra_args+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
fi

if train_supports_arg "--amp" && [ "${#amp_args[@]}" -gt 0 ]; then
    train_extra_args+=("${amp_args[@]}")
fi

# 如果训练语料还没准备好，就先用原始数据生成一次。
if [ ! -f "$DATA_FILE" ]; then
    if [ -d "dataset" ]; then
        "$PYTHON_CMD" data_process.py
    else
        echo "未找到训练数据目录：dataset"
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

# 默认按服务器环境启动：禁用实时图窗，额外参数可继续从命令行透传。
"$PYTHON_CMD" train.py \
    --data "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --device auto \
    --batch-size "$train_batch_size" \
    --no-live-plot \
    "${train_extra_args[@]}" \
    "$@"
