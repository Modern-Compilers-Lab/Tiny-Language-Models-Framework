#!/bin/bash
# ============================================================================
#  Parallel ZERO-SHOT variable-renaming evaluation (8 GPUs)
# ============================================================================
#  No fine-tuning. Raw pre-trained encoder -> mean-pooled cosine similarity
#  on POJ-104 -> top-1 retention under the shared var_xxx renamed test sets.
#
#  Same 14 models as the fine-tuned run:
#    - baseline_codebert     : microsoft/codebert-base  (via local safetensors)
#    - baseline_contrabert   : claudios/ContraBERT_C    (from HF hub cache)
#    - 12 exp1_diverse checkpoints every 9 epochs: 9, 18, 27, ..., 108
#
#  GPU policy: exactly one process per GPU; refill on completion. Never split.
#
#  Usage:  bash run_zeroshot_parallel.sh
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/ia2921/anaconda3/envs/codebert_varname/bin/python}"
CHECKPOINT_DIR="checkpoints/exp1_diverse"
DATA_DIR="data_cache/poj104"
EVAL_DIR="eval_results_varname"
LOG_DIR="logs"
GPUS=(0 1 2 3 4 5 6 7)
STEPS_PER_EPOCH=918
EPOCHS=(9 18 27 36 45 54 63 72 81 90 99 108)

COMMON="--data_dir ${DATA_DIR} --renamed_dir ${DATA_DIR} --output_dir ${EVAL_DIR} \
  --block_size 400 --embedding meanpool --seed 123456"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

# Preflight: shared renamed sets must exist (built during the fine-tuned run).
for n in 1 4 8; do
    if [ ! -f "${DATA_DIR}/test_renamed_N${n}.jsonl" ]; then
        echo "Missing ${DATA_DIR}/test_renamed_N${n}.jsonl -> regenerating..."
        "$PYTHON" generate_renamed_testsets.py \
            --data_dir "${DATA_DIR}" --rename_counts 1,4,8 --seed 123456
        break
    fi
done

# Job list:  "experiment_name|checkpoint"
# IMPORTANT: both baselines are explicitly included.
JOBS=()
JOBS+=("baseline_codebert|models_cache/codebert_base_st")      # CodeBERT baseline (safetensors)
JOBS+=("baseline_contrabert|claudios/ContraBERT_C")            # ContraBERT_C baseline
for e in "${EPOCHS[@]}"; do
    step=$(( e * STEPS_PER_EPOCH ))
    JOBS+=("epoch_${e}|${CHECKPOINT_DIR}/checkpoint-${step}")
done
NJOBS=${#JOBS[@]}

echo "============================================================"
echo "  Zero-shot evaluation: ${NJOBS} jobs across ${#GPUS[@]} GPUs"
echo "  Models: 2 baselines + ${#EPOCHS[@]} checkpoints"
echo "  Output: ${EVAL_DIR}/zeroshot_<name>.json   Logs: ${LOG_DIR}/zeroshot_<name>.log"
echo "============================================================"

run_job() {
    local gpu="$1" name="$2" ckpt="$3"
    local log="${LOG_DIR}/zeroshot_${name}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON" evaluate_zeroshot.py \
        --checkpoint "${ckpt}" --experiment_name "${name}" ${COMMON} \
        > "${log}" 2>&1
}

declare -A PID2GPU
FREE_GPUS=("${GPUS[@]}")
ji=0

while [ $ji -lt $NJOBS ] || [ ${#PID2GPU[@]} -gt 0 ]; do
    while [ ${#FREE_GPUS[@]} -gt 0 ] && [ $ji -lt $NJOBS ]; do
        last=$(( ${#FREE_GPUS[@]} - 1 ))
        gpu="${FREE_GPUS[$last]}"
        unset 'FREE_GPUS[$last]'
        FREE_GPUS=("${FREE_GPUS[@]}")

        IFS='|' read -r name ckpt <<< "${JOBS[$ji]}"
        echo "[$(date +%H:%M:%S)] GPU ${gpu} <- ${name}  ($((ji+1))/${NJOBS})  ${ckpt}"
        run_job "$gpu" "$name" "$ckpt" &
        PID2GPU[$!]=$gpu
        ji=$((ji+1))
    done

    wait -n 2>/dev/null || true
    for pid in "${!PID2GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$?
            gpu="${PID2GPU[$pid]}"
            unset 'PID2GPU[$pid]'
            FREE_GPUS+=("$gpu")
            echo "[$(date +%H:%M:%S)] GPU ${gpu} freed (pid ${pid} exit ${rc})"
        fi
    done
done

echo ""
echo "============================================================"
echo "  All ${NJOBS} zero-shot jobs finished. Building table..."
echo "============================================================"
"$PYTHON" build_zeroshot_table.py --eval_dir "${EVAL_DIR}"
