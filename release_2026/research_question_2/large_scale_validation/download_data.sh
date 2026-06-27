#!/bin/bash
# Download CodeSearchNet and POJ-104 datasets.
# Usage: bash download_data.sh [data_dir]
#
# CodeSearchNet: 1,880,853 training examples across 6 languages
# POJ-104: Clone detection dataset (12,000 test samples) for evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/data_cache}"

echo "============================================================"
echo "  Downloading datasets to: ${DATA_DIR}"
echo "============================================================"

mkdir -p "${DATA_DIR}"

# Download CodeSearchNet (cached by HuggingFace datasets library)
echo ""
echo "Downloading CodeSearchNet (all 6 languages)..."
echo "  This downloads ~2GB and may take several minutes on first run."
python "${SCRIPT_DIR}/download_data.py" "${DATA_DIR}"

# Verify POJ-104
POJ_DIR="${DATA_DIR}/poj104"
for f in train.jsonl valid.jsonl test.jsonl; do
    if [ ! -f "${POJ_DIR}/${f}" ]; then
        echo "ERROR: ${POJ_DIR}/${f} not found after download."
        echo "Please manually download POJ-104 from:"
        echo "  https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104"
        echo "Place train.jsonl, valid.jsonl, test.jsonl in: ${POJ_DIR}/"
        exit 1
    fi
done

echo ""
echo "Verifying data..."
TRAIN_COUNT=$(wc -l < "${POJ_DIR}/train.jsonl")
TEST_COUNT=$(wc -l < "${POJ_DIR}/test.jsonl")
echo "  POJ-104 train: ${TRAIN_COUNT} examples"
echo "  POJ-104 test:  ${TEST_COUNT} examples"

echo ""
echo "============================================================"
echo "  Data download complete."
echo "============================================================"
