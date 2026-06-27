import os
import subprocess
import time
import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
percentages = [1]

checkpoints = [
    "checkpoint_0.03.pth", "checkpoint_0.07.pth", "checkpoint_0.10.pth",
    "checkpoint_0.13.pth", "checkpoint_0.17.pth", "checkpoint_0.20.pth",
    "checkpoint_0.23.pth", "checkpoint_0.27.pth", "checkpoint_0.30.pth",
    "checkpoint_0.33.pth", "checkpoint_0.37.pth", "checkpoint_0.40.pth",
    "checkpoint_0.43.pth", "checkpoint_0.47.pth", "checkpoint_0.50.pth",
    "checkpoint_0.53.pth", "checkpoint_0.57.pth", "checkpoint_0.60.pth",
    "checkpoint_0.63.pth", "checkpoint_0.67.pth", "checkpoint_0.70.pth",
    "checkpoint_0.73.pth", "checkpoint_0.77.pth", "checkpoint_0.80.pth",
    "checkpoint_0.83.pth", "checkpoint_0.87.pth", "checkpoint_0.90.pth",
    "checkpoint_0.93.pth", "checkpoint_0.97.pth", "checkpoint_1.00.pth",
]

BASE         = "/data/ia2921/Tiny_language_model_framework"
EVAL_SCRIPT  = f"{BASE}/3Evaluations/experiment_reboot/exp2_alpha/eval.py"
OUTPUT_PATH  = f"{BASE}/3Evaluations/experiment_reboot/results_semantic_preserving_edits"
LOG_PATH     = f"{BASE}/3Evaluations/experiment_reboot/logs"
GPUS         = [0, 1, 2, 3,4,5,6,7]
USE_BUSY_GPUS = False
POLL_INTERVAL = 15  # seconds

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def pct_str(p):
    """Format percentage for use in directory names (e.g. 0.01, 10, 50)."""
    return str(p)   # already matches the folder naming convention

def gpu_free(gpu_id, threshold=30000):
    out = subprocess.getoutput(
        f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
    )
    try:
        return (int(out.strip()) < threshold) or USE_BUSY_GPUS
    except ValueError:
        return False

# ──────────────────────────────────────────────
# BUILD JOB LIST
# Each job: (ckpt_path, ckpt_name, pct, data_dir, test_path)
# ──────────────────────────────────────────────
jobs = []
for p in percentages:
    ps = pct_str(p)
    data_dir  = f"/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/alpha_0.X/300m"
    for ckpt in checkpoints:
        test_path = f"/data/ia2921/Tiny_language_model_framework/3Evaluations/experiment_reboot/exp2_alpha/semantic_preserving_edits_files/eval_jobs1_{ckpt}_ID/test.txt"
        ckpt_path = f"/data/ia2921/Tiny_language_model_framework/2Training/experiment_reboot/exp2_alpha/checkpoints1/{ckpt}"
        jobs.append((ckpt_path, ckpt, ps, data_dir, test_path))

print(f"Total jobs: {len(jobs)}  ({len(percentages)} percentages × {len(checkpoints)} checkpoints)")

# Optional sanity-check: warn about missing checkpoint files
missing = [j[0] for j in jobs if not os.path.exists(j[0])]
if missing:
    print(f"WARNING: {len(missing)} checkpoint files not found on disk:")
    for m in missing:
        print(f"  {m}")

# ──────────────────────────────────────────────
# SCHEDULER LOOP
# ──────────────────────────────────────────────
while jobs:
    for gpu in GPUS:
        if not jobs:
            break
        if gpu_free(gpu):
            ckpt_path, ckpt_name, pct, data_dir, test_path = jobs.pop(0)

            job_id    = f"eval_{ckpt_name}_p_{pct}"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screen_name = f"{job_id}_{timestamp}"

            out_dir  = os.path.join(OUTPUT_PATH, job_id)
            log_file = os.path.join(LOG_PATH, f"{screen_name}.log")

            print(f"[{timestamp}] GPU {gpu} → {ckpt_name} | p={pct}")

            cmd = (
                f"screen -dmS {screen_name} bash -c '"
                f"source /home/ia2921/anaconda3/etc/profile.d/conda.sh && "
                f"conda activate torch_env && "
                f"CUDA_VISIBLE_DEVICES={gpu} python \"{EVAL_SCRIPT}\" "
                f"--cptpth \"{ckpt_path}\" "
                f"--datpth \"{data_dir}/\" "
                f"--tstpth \"{test_path}\" "
                f"--outpth \"{out_dir}\" "
                f"> \"{log_file}\" 2>&1; exit'"
            )
            subprocess.Popen(cmd, shell=True)

    time.sleep(POLL_INTERVAL)

print("All jobs dispatched.")