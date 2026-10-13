#!/usr/bin/env bash
# Rebuild from ORIGINAL researcher-supplied environment locks. No training.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS=(logs pka logpapp density)
if ! command -v conda >/dev/null 2>&1; then
  echo 'ERROR: 找不到 conda。请先在目标 Linux 服务器安装 Miniforge/Miniconda，然后重新运行。' >&2
  exit 2
fi

want_model() {
  local name="$1"
  if [ "$#" -eq 0 ]; then return 0; fi
  return 0
}
if [ "$#" -gt 0 ]; then MODELS=("$@"); fi

for MODEL in "${MODELS[@]}"; do
  case "$MODEL" in
    logs|pka|logpapp) PYTHON_VERSION=3.11.11 ;;
    density) PYTHON_VERSION=3.12.13 ;;
    logp) echo 'LogP 是 Python 3.5.6 / TF1.14 / DeepChem2.2 旧环境：请使用已经验证的旧 Linux 环境，并设置 LOGP_PYTHON。不会自动替换。' >&2; continue;;
    *) echo "未知模型: $MODEL" >&2; exit 2;;
  esac
  PREFIX="$ROOT/.venvs/$MODEL"
  echo "[1/3] 创建/检查 $MODEL 环境: python=$PYTHON_VERSION"
  if [ ! -x "$PREFIX/bin/python" ]; then
     conda create --yes --prefix "$PREFIX" "python=$PYTHON_VERSION" pip
  fi
  ACTUAL=$("$PREFIX/bin/python" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')
  if [ "$ACTUAL" != "$PYTHON_VERSION" ]; then
    echo "错误：$MODEL Python 版本 $ACTUAL != $PYTHON_VERSION，停止以免混装" >&2
    exit 3
  fi
  echo "[2/3] 严格安装 $MODEL 原包锁定依赖"
  "$PREFIX/bin/python" -m pip install --disable-pip-version-check --requirement "$ROOT/envs/requirements_${MODEL}_exact.txt"
  if [ "$MODEL" = density ]; then
    echo '[3/3] 安装原 Density PyTorch GPU 版本 torch 2.10.0+cu128 (需要 CUDA 兼容的 Linux GPU)'
    "$PREFIX/bin/python" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
  else
    echo "[3/3] 完成 $MODEL"
  fi
  echo "[installed] $MODEL => $PREFIX/bin/python"
done
echo '安装流程结束。建议检查各模型：'
echo "  python '$ROOT/scripts/match_and_run_v06.py' --verify-only"
echo '若要启用旧 LogP: LOGP_PYTHON=/path/to/working/python3.5'
echo '注：脚本不覆盖已有环境；原有环境一旦不符合版本要求，立即停止。'
