#!/bin/bash
# ============================================================================
#  Parallel re-evaluation of variable-renaming robustness (8 GPUs)
# ============================================================================
#
#  Re-runs the FINE-TUNED evaluation only, for:
#    - 2 baselines: microsoft/codebert-base, claudios/ContraBERT_C
#    - exp1_diverse checkpoints every 9 epochs: 9,18,27,...,108
#
#  Each job (one per model): load checkpoint -> fine-tune 2 epochs on POJ-104
#  -> MAP@R on the original test set -> accuracy on the SHARED var_xxx renamed
#  sets (N=1,4,8).
#
#  GPU policy: exactly one job per GPU at a time. A GPU is refilled as soon as
#  its job finishes. A single process is never split across GPUs (no
#  DataParallel; pinned via CUDA_VISIBLE_DEVICES).
#
#  Usage:
#    bash run_eval_parallel.sh
#    PYTHON=/path/to/python bash run_eval_parallel.sh   # override interpreter
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
RENAME_COUNTS="0,1,4,8"

FT_COMMON="--data_dir ${DATA_DIR} --renamed_dir ${DATA_DIR} \
  --num_epochs 2 --train_batch_size 8 --eval_batch_size 16 \
  --learning_rate 2e-5 --seed 123456 --block_size 400 \
  --rename_counts ${RENAME_COUNTS} --output_dir ${EVAL_DIR}"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

# ── Preflight: ensure shared renamed test sets exist (generate once if not) ──
NEED_GEN=""
for n in 1 4 8; do
    [ -f "${DATA_DIR}/test_renamed_N${n}.jsonl" ] || NEED_GEN="yes"
done
if [ -n "$NEED_GEN" ]; then
    echo "Shared renamed test sets missing -> generating (fixed seed)..."
    "$PYTHON" generate_renamed_testsets.py \
        --data_dir "${DATA_DIR}" --rename_counts 1,4,8 --seed 123456
    echo ""
fi

# ── Build job list:  "name|checkpoint" ──
JOBS=()
JOBS+=("baseline_codebert|microsoft/codebert-base")
JOBS+=("baseline_contrabert|claudios/ContraBERT_C")
for e in "${EPOCHS[@]}"; do
    step=$(( e * STEPS_PER_EPOCH ))
    JOBS+=("finetuned_epoch_${e}|${CHECKPOINT_DIR}/checkpoint-${step}")
done
NJOBS=${#JOBS[@]}

echo "============================================================"
echo "  Parallel evaluation: ${NJOBS} jobs over ${#GPUS[@]} GPUs"
echo "  Output:   ${EVAL_DIR}/   Logs: ${LOG_DIR}/eval_<name>.log"
echo "  Python:   ${PYTHON}"
echo "============================================================"

run_job() {
    local gpu="$1" name="$2" ckpt="$3"
    local log="${LOG_DIR}/eval_${name}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON" evaluate_robustness.py \
        --checkpoint "${ckpt}" --experiment_name "${name}" ${FT_COMMON} \
        > "${log}" 2>&1
}

declare -A PID2GPU
FREE_GPUS=("${GPUS[@]}")
ji=0

while [ $ji -lt $NJOBS ] || [ ${#PID2GPU[@]} -gt 0 ]; do
    # Fill every free GPU with a pending job.
    while [ ${#FREE_GPUS[@]} -gt 0 ] && [ $ji -lt $NJOBS ]; do
        last=$(( ${#FREE_GPUS[@]} - 1 ))
        gpu="${FREE_GPUS[$last]}"
        unset 'FREE_GPUS[$last]'
        FREE_GPUS=("${FREE_GPUS[@]}")          # reindex

        IFS='|' read -r name ckpt <<< "${JOBS[$ji]}"
        echo "[$(date +%H:%M:%S)] GPU ${gpu} <- ${name}  ($((ji+1))/${NJOBS})  ${ckpt}"
        run_job "$gpu" "$name" "$ckpt" &
        PID2GPU[$!]=$gpu
        ji=$((ji+1))
    done

    # Wait for at least one job to finish, then reap all finished jobs.
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
echo "  All ${NJOBS} jobs finished. Building summary table..."
echo "============================================================"
"$PYTHON" build_table.py --eval_dir "${EVAL_DIR}"
