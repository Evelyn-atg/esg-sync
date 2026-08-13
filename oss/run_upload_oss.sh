#!/bin/bash
# ============================================================
# ESG OCR -> 阿里云 OSS 上传脚本（集群运行）
#
# 用法:
#   bash run_upload_oss.sh                # 全量上传
#   bash run_upload_oss.sh --resume       # 断点续传
#   bash run_upload_oss.sh --dry-run      # 只扫描
#   bash run_upload_oss.sh --verify       # 校验
#   bash run_upload_oss.sh --workers 16   # 16线程
#
# 首次运行会自动安装 oss2 依赖。
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ESG OCR -> OSS Uploader"
echo "  Script dir: $SCRIPT_DIR"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- 集群环境：尝试加载 anaconda module（本地无 module 命令时自动跳过）---
if command -v module &>/dev/null 2>&1; then
    module load apps/anaconda3/2021.05 2>/dev/null || true
fi

# --- 选 Python3 ---
PYTHON=""
for candidate in python3 /opt/anaconda3/bin/python3 ~/anaconda3/bin/python3 ~/miniconda3/bin/python3 /usr/bin/python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python3 not found. Please install or activate conda."
    exit 1
fi

echo "[INFO] Using Python: $PYTHON"
$PYTHON --version

# --- 安装依赖 ---
echo ""
echo "[INFO] Checking dependencies..."

# oss2
if ! $PYTHON -c "import oss2" 2>/dev/null; then
    echo "[INFO] Installing oss2..."
    $PYTHON -m pip install oss2 --quiet
    echo "[INFO] oss2 installed."
else
    echo "[INFO] oss2 already installed."
fi

# python-dotenv (optional, for .env auto-loading)
if ! $PYTHON -c "import dotenv" 2>/dev/null; then
    echo "[INFO] Installing python-dotenv..."
    $PYTHON -m pip install python-dotenv --quiet 2>/dev/null || true
    echo "[INFO] python-dotenv installed (or skipped)."
else
    echo "[INFO] python-dotenv already installed."
fi

# --- 检查/生成 .env ---
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.b64" ]; then
        echo "[INFO] .env not found, decoding from .env.b64 ..."
        base64 -d "$SCRIPT_DIR/.env.b64" > "$SCRIPT_DIR/.env"
        echo "[INFO] .env created from .env.b64"
    else
        echo "[ERROR] .env file not found in $SCRIPT_DIR"
        echo "        Please create it with OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET,"
        echo "        OSS_ENDPOINT, OSS_BUCKET_NAME"
        exit 1
    fi
fi

echo ""
echo "[INFO] .env ready. Starting upload..."
echo ""

# --- 运行 ---
# 透传所有命令行参数
$PYTHON "$SCRIPT_DIR/upload_ocr_to_oss.py" "$@"

echo ""
echo "============================================================"
echo "  Done. $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
