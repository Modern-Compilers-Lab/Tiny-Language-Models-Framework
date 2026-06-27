#!/bin/bash
# Train CodeBERT with diverse variable names using multi-GPU DDP.
#
# Usage:
#   bash train.sh                          # 8 GPUs, 100K steps (~109 epochs)
#   bash train.sh --num_gpus 4             # 4 GPUs
#   bash train.sh --num_gpus 2 --max_steps 1000   # Quick test
#   bash train.sh --resume                 # Resume from latest checkpoint
#
# Parameters:
#   --num_gpus N              Number of GPUs (default: 8)
#   --max_steps N             Total training steps (default: 100000, matching CodeBERT paper)
#   --save_every_n_epochs N   Save checkpoint every N epochs (default: 10)
#   --output_dir DIR          Checkpoint directory (default: checkpoints/exp1_diverse)
#   --resume                  Resume from latest checkpoint in output_dir
#   --data_dir DIR            Data directory (default: data_cache)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Parse arguments ──
NUM_GPUS=8
MAX_STEPS=100000
SAVE_EVERY_N_EPOCHS=1
OUTPUT_DIR="checkpoints/exp1_diverse"
DATA_DIR="data_cache"
RESUME=""
PARALLEL_EVAL_GPUS=""
PARALLEL_EVAL_EVERY_N=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)  NUM_GPUS="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --save_every_n_epochs) SAVE_EVERY_N_EPOCHS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --resume)    RESUME="yes"; shift ;;
        --parallel_eval_gpus)    PARALLEL_EVAL_GPUS="$2"; shift 2 ;;
        --parallel_eval_every_n) PARALLEL_EVAL_EVERY_N="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Detect GPUs ──
DETECTED_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$DETECTED_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi
if [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
    echo "WARNING: Requested ${NUM_GPUS} GPUs but only ${DETECTED_GPUS} available. Using ${DETECTED_GPUS}."
    NUM_GPUS=$DETECTED_GPUS
fi

# ── Detect GPU memory for batch size ──
GPU_MEM=$(python -c "import torch; p=torch.cuda.get_device_properties(0); print(getattr(p,'total_memory',getattr(p,'total_mem',0))//1024**3)" 2>/dev/null || echo 40)
if [ "$GPU_MEM" -ge 50 ]; then
    PER_GPU_BATCH=64
else
    PER_GPU_BATCH=32
fi

# ── Effective batch = 2048 (matching CodeBERT paper) ──
EFFECTIVE_BATCH=2048
GRAD_ACCUM=$((EFFECTIVE_BATCH / (NUM_GPUS * PER_GPU_BATCH)))
if [ "$GRAD_ACCUM" -lt 1 ]; then
    GRAD_ACCUM=1
fi
ACTUAL_EFFECTIVE=$((NUM_GPUS * PER_GPU_BATCH * GRAD_ACCUM))

# ── Resume from latest checkpoint ──
RESUME_ARG=""
if [ "$RESUME" = "yes" ] && [ -d "${OUTPUT_DIR}" ]; then
    LATEST=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -n "$LATEST" ] && [ -d "${OUTPUT_DIR}/checkpoint-${LATEST}" ]; then
        RESUME_ARG="--resume_from ${OUTPUT_DIR}/checkpoint-${LATEST}"
        echo "Resuming from: ${OUTPUT_DIR}/checkpoint-${LATEST}"
    fi
fi

# ── Compute epochs ──
# CodeSearchNet: 1,880,853 examples
# Steps per epoch = 1880853 / effective_batch
STEPS_PER_EPOCH=$((1880853 / ACTUAL_EFFECTIVE))
TOTAL_EPOCHS=$((MAX_STEPS / STEPS_PER_EPOCH))

# ── Print configuration ──
echo ""
echo "============================================================"
echo "  CodeBERT Training with Diverse Variable Names"
echo "============================================================"
echo ""
echo "  GPUs:              ${NUM_GPUS} x $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)"
echo "  GPU memory:        ${GPU_MEM} GB"
echo "  Per-GPU batch:     ${PER_GPU_BATCH}"
echo "  Grad accumulation: ${GRAD_ACCUM}"
echo "  Effective batch:   ${ACTUAL_EFFECTIVE}"
echo "  Learning rate:     5e-4 (warmup 10K steps, linear decay)"
echo "  Max steps:         ${MAX_STEPS}"
echo "  Steps per epoch:   ${STEPS_PER_EPOCH}"
echo "  Total epochs:      ~${TOTAL_EPOCHS}"
echo "  Max seq length:    512"
echo "  FP16:              yes"
echo "  Diversify vars:    yes (re-randomized each epoch)"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Data:              ${DATA_DIR}"
echo ""
echo "  Save checkpoint:   every ${SAVE_EVERY_N_EPOCHS} epochs + final"
echo "  Zero-shot eval:    after each epoch"
echo ""
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}" logs

# ── Pick a random master port to avoid conflicts ──
MASTER_PORT=$((29500 + RANDOM % 500))

# ── Launch training ──
# Use python -m torch.distributed.run instead of torchrun to ensure
# the correct Python environment is used.
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train_codebert.py \
    --per_gpu_batch_size ${PER_GPU_BATCH} \
    --effective_batch_size ${EFFECTIVE_BATCH} \
    --lr 5e-4 \
    --warmup_steps 10000 \
    --max_steps ${MAX_STEPS} \
    --max_length 512 \
    --data_pct 1.0 \
    --diversify_varnames \
    --eval_data_dir "${DATA_DIR}/poj104" \
    --output_dir "${OUTPUT_DIR}" \
    --log_interval 50 \
    --save_interval 999999 \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --parallel_eval_gpus "${PARALLEL_EVAL_GPUS}" \
    --parallel_eval_every_n ${PARALLEL_EVAL_EVERY_N} \
    --seed 42 \
    ${RESUME_ARG} \
    2>&1 | tee logs/training.log

echo ""
echo "============================================================"
echo "  Training complete."
echo "  Checkpoints saved to: ${OUTPUT_DIR}"
echo "  Log saved to: logs/training.log"
echo "============================================================"
