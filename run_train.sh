#!/bin/bash
# Reward Model 训练启动脚本
# 用法: bash run_train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/ssd/cyfu/miniconda3/envs/diff/bin/python"

echo "=========================================="
echo "  Reward Model Latent Training"
echo "  Dir: ${SCRIPT_DIR}"
echo "  Python: ${PYTHON}"
echo "=========================================="

cd "${SCRIPT_DIR}"
${PYTHON} train.py "$@"
