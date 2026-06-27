"""
CUDA_VISIBLE_DEVICES=3 python /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/simple_arch.py\
  --cptpth /data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train/checkpoints1_10/checkpoint_0.21.pth \
  --datpth /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_10_p/ \
  --tstpth /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files_new/eval_jobs1_checkpoint_0.21.pth_ID_p_10/test.txt \
  --outpth /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/hidden_variables
"""
# /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/OOD/data_10_p/hidden_variables/data/test.txt
# /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files/eval_jobs1_checkpoint_0.21.pth_ID_p_10/test.txt


# _________________________________SCRIPT_________________________________________________

# cd /Path/to/folder/where/this/script/lives
# python scheduler.py
# you can customize this script to suite your specific need , for example you might have many test files, or checkpoitns from multiple folders ..etc

#_____________________________IMPORTANT_HYPERPARAMETERS___________________________________

#use ctrl+f for quick access:
# there is not really any high-level variables to tweak from here
# evaluation_script_path : absolute file path to evaluation script
# checkpoints : file names of all checkpoints to evaluate (must be all in a single folder)
# gpus : list all gpus that the scheduler can use to launch evaluation, more gpus means more processes will be launched at once and therefore less time consumed overall
# pth_to_checkpoints : absolute folder path to checkpoints you want to evaluate
# pth_to_test : absolute file path to test file
# output_path : absolute folder path to where you want to store results
# threshold (gpu_free() method) : a memory limit that protects the gpus from recieving too many processes at once, you can test multiple values to see which fits best
# usually a limit of 4 processes is good (equivalent to maybe 20000 MB), just make sure you have a limit when you know the scheduler will produce a lot of jobs


#_____________________________________CODE_______________________________________________



import os


evaluation_script_path = ["/data/ia2921/release_2026/dry_framework/evaluation/eval.py"]
checkpoints = [
 "checkpoint_0.04.pth","checkpoint_0.08.pth","checkpoint_0.12.pth","checkpoint_0.17.pth",
 "checkpoint_0.21.pth","checkpoint_0.25.pth","checkpoint_0.29.pth","checkpoint_0.33.pth"
]

pth_to_checkpoints = "/data/ia2921/release_2026/dry_framework/training/checkpoints1"
pth_to_test = "/data/ia2921/release_2026/dry_framework/data_generation/data/test.txt"
pth_to_data = "/data/ia2921/release_2026/dry_framework/data_generation/data/"

#alpha/beta/tokenizer
gpus = [0,1,2,3]
tests = []
full_checkpoint_path = []
full_checkpoint_path_rand_enc = []

tmp = []
for chk in checkpoints :
    tmp.append(pth_to_checkpoints+"/"+chk)
tests.append(  [pth_to_test] )
full_checkpoint_path.append(tmp)


for arr in full_checkpoint_path :
    for path in arr:
        print(os.path.exists(path))




test_datasets_titles = ["ID"]

#alpha/beta/tokenizer
output_path = "/data/ia2921/release_2026/dry_framework/evaluation/results"
use_busy_gpus = False
#best-model.pth
#joined_path = os.path.join(some_path, a_string)

import subprocess

def gpu_free(gpu_id, threshold=1000):  # MB
    out = subprocess.getoutput(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}")
    return (int(out.strip()) < threshold) or use_busy_gpus

# tests
# full_checkpoint_path
# full_checkpoint_path_rand_enc

jobs1= []

test_datasets = list(zip(tests[0], test_datasets_titles))
check_couple = list(zip(full_checkpoint_path[0], checkpoints))
jobs1.extend([(checkpoint,check_name, dataset_pth, dataset_title) for checkpoint,check_name in check_couple for dataset_pth, dataset_title in test_datasets])




# with open("/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/output.txt", "w") as f:
#     for job in jobs1:
#         print(job, file=f)
#     for job in jobs2:
#         print(job, file=f)

# print(len(jobs1),len(jobs2))




import time
import datetime
while jobs1:
    for gpu in gpus:
        if not jobs1:
            break
        if gpu_free(gpu):
            print(f"gpu number {gpu} is free, i'll use it")
            ckpt_path,ckpt, test_pth, test_title = jobs1.pop(0)
            print(f"evaluating {ckpt_path} on {test_title}")
            checkpoint_name = ckpt
            #full_path = ckpt_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subprocess.Popen(f"screen -dmS eval_job1_{ckpt}_{test_title}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[0]}\" --cptpth \"{ckpt_path}\" --datpth \"{pth_to_data}\" --tstpth \"{test_pth}\" --outpth \"{os.path.join(output_path, checkpoint_name)}\"; exit'", shell=True)
    time.sleep(30)  # poll every 30s

