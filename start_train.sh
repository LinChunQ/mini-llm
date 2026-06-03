#!/usr/bin/env bash
set -euo pipefail

# 始终切到仓库根目录，确保从任意路径执行都能找到项目文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TMPDIR="${TMPDIR:-$SCRIPT_DIR/.tmp}"
mkdir -p "$TMPDIR"
export TMPDIR

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATA_FILE="${DATA_FILE:-dataset/poetry_train.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
USE_VENV="${USE_VENV:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-10}"

PYTHON_CMD="$PYTHON_BIN"
if [ "$USE_VENV" = "1" ]; then
    # 先创建虚拟环境，再安装依赖，避免污染系统 Python。
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    PYTHON_CMD="$VENV_DIR/bin/python"
fi

# 如果当前环境已经有训练所需依赖，就不重复安装，避免无谓占用磁盘。
if ! "$PYTHON_CMD" - <<'PY' >/dev/null 2>&1
import torch
import matplotlib
import tqdm
PY
then
    # 网络较慢时，使用镜像源和更长超时，降低依赖安装失败概率。
    "$PYTHON_CMD" -m pip install --upgrade pip --timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" \
        --no-cache-dir --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
    "$PYTHON_CMD" -m pip install -r requirements.txt --timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" \
        --no-cache-dir --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
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

# 默认按服务器环境启动：禁用实时图窗，参数也可以在脚本后继续追加覆盖。
"$PYTHON_CMD" train.py \
    --data "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --device auto \
    --no-live-plot \
    "$@"
