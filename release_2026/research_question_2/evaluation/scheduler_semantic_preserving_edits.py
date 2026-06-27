"""
CUDA_VISIBLE_DEVICES=3 python /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/rng_pos_end.py\
  --cptpth /data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train/checkpoints1_0.01/checkpoint_1.00.pth \
  --datpth /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_0.01_p/ \
  --tstpth /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_0.01_p/test.txt \
  --outpth /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/results
"""

import os


#alpha/beta/tokenizer
#data_dir_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/gamma_1_X_gen/data_200/"
#on every architecture
#checkpoint_path = "/data/ia2921/Tiny_language_model_framework/2Training/gamma_200_alibi_cb/checkpoints1/"
#on every architecture
evaluation_script_path = ["/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/simple_arch.py","/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/simple_arch.py"]
checkpoints = [
 "checkpoint_0.04.pth","checkpoint_0.08.pth","checkpoint_0.12.pth","checkpoint_0.17.pth",
 "checkpoint_0.21.pth","checkpoint_0.25.pth","checkpoint_0.29.pth","checkpoint_0.33.pth",
 "checkpoint_0.37.pth","checkpoint_0.42.pth","checkpoint_0.46.pth","checkpoint_0.50.pth",
 "checkpoint_0.54.pth","checkpoint_0.58.pth","checkpoint_0.62.pth","checkpoint_0.67.pth",
 "checkpoint_0.71.pth","checkpoint_0.75.pth","checkpoint_0.79.pth","checkpoint_0.83.pth",
 "checkpoint_0.87.pth","checkpoint_0.92.pth","checkpoint_0.96.pth","checkpoint_1.00.pth"
]

#alpha/beta/tokenizer

percentages = ["10","1","0.1","0.01"]
tests = []
full_checkpoint_path = []
full_checkpoint_path_rand_enc = []
for p in percentages : 
    tmp = []
    tmp_rand_enc = []
    for chk in checkpoints :
        tmp.append("/data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train/checkpoints1_"+p+"/"+chk)
        tmp_rand_enc.append("/data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train_random_pos/checkpoints1_"+p+"/"+chk)
    tests.append(  ["/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_"+p+"_p/test.txt"] )
    full_checkpoint_path.append(tmp)
    full_checkpoint_path_rand_enc.append(tmp_rand_enc)

for arr in full_checkpoint_path :
    for path in arr:
        print(os.path.exists(path))




test_datasets_titles = ["semantic_preserving_ID"]

#alpha/beta/tokenizer
output_path = "/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/results"
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
jobs2= []
for i in range(len(percentages)):
    test_datasets = list(zip(tests[i], test_datasets_titles))
    check_couple = list(zip(full_checkpoint_path[i], checkpoints,[percentages[i]]*len(full_checkpoint_path[i])))
    jobs1.extend([(checkpoint,check_name,percentage, dataset_pth, dataset_title) for checkpoint,check_name,percentage in check_couple for dataset_pth, dataset_title in test_datasets])
for i in range(len(percentages)):
    test_datasets = list(zip(tests[i], test_datasets_titles))
    check_couple = list(zip(full_checkpoint_path_rand_enc[i], checkpoints,[percentages[i]]*len(full_checkpoint_path_rand_enc[i])))
    jobs2.extend([(checkpoint,check_name,percentage, dataset_pth, dataset_title) for checkpoint,check_name,percentage in check_couple for dataset_pth, dataset_title in test_datasets])
gpus = [0,1,2,3]

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
            ckpt_path,ckpt,percentage, test_pth, test_title = jobs1.pop(0)
            print(f"evaluating {ckpt_path} on {test_title}")
            checkpoint_name = f"eval_jobs1_{ckpt}_{test_title}_p_{percentage}"
            #full_path = ckpt_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subprocess.Popen(f"screen -dmS eval_job1_{ckpt}_{test_title}_p_{percentage}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[0]}\" --cptpth \"{ckpt_path}\" --datpth \"/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_{percentage}_p/\" --tstpth \"/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files_new/eval_jobs1_{ckpt}_ID_p_{percentage}/test.txt\" --outpth \"{os.path.join(output_path, checkpoint_name)}\"; exit'", shell=True)
    time.sleep(30)  # poll every 30s


while jobs2:
    for gpu in gpus:
        if not jobs2:
            break
        if gpu_free(gpu):
            print(f"gpu number {gpu} is free, i'll use it")
            ckpt_path,ckpt,percentage, test_pth, test_title = jobs2.pop(0)
            print(f"evaluating {ckpt_path} on {test_title}")
            checkpoint_name = f"eval_jobs2_{ckpt}_{test_title}_p_{percentage}"
            #full_path = ckpt_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subprocess.Popen(f"screen -dmS eval_job2_{ckpt}_{test_title}_p_{percentage}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[1]}\" --cptpth \"{ckpt_path}\" --datpth \"/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_{percentage}_p/\" --tstpth \"/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files/eval_jobs2_{ckpt}_ID_p_{percentage}/test.txt\" --outpth \"{os.path.join(output_path, checkpoint_name)}\"; exit'", shell=True)
    time.sleep(30)  # poll every 30s
