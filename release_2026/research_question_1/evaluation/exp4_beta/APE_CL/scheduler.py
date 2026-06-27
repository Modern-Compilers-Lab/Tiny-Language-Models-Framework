"""
CUDA_VISIBLE_DEVICES=3 python /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/simple_arch.py\
  --cptpth /data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train/checkpoints1_10/checkpoint_0.21.pth \
  --datpth /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_10_p/ \
  --tstpth /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files_new/eval_jobs1_checkpoint_0.21.pth_ID_p_10/test.txt \
  --outpth /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/hidden_variables
"""
# /data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/OOD/data_10_p/hidden_variables/data/test.txt
# /data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/semantic_preserving_edits_files/eval_jobs1_checkpoint_0.21.pth_ID_p_10/test.txt

import os


#alpha/beta/tokenizer
#data_dir_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/gamma_1_X_gen/data_200/"
#on every architecture

#on every architecture
gpus = [5,6]
data_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/beta_0.X/20m"
checkpoint_path = "/data/ia2921/Tiny_language_model_framework/2Training/experiment_reboot/exp4_beta/exp4_beta_APE/checkpoints1"
ood_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/beta_0.X/OOD"
evaluation_script_path = ["/data/ia2921/Tiny_language_model_framework/3Evaluations/experiment_reboot/exp4_beta/APE_CL/optimus_eval.py"]
# checkpoints = [
#  'checkpoint_0.10.pth',
#  'checkpoint_0.20.pth',
#  'checkpoint_0.30.pth',
#  'checkpoint_0.40.pth',
#  'checkpoint_0.50.pth',
#  'checkpoint_0.60.pth',
#  'checkpoint_0.70.pth',
#  'checkpoint_0.80.pth',
#  'checkpoint_0.90.pth',
#  'checkpoint_1.00.pth'
# ]

checkpoints = ["best-model.pth"]
#alpha/beta/tokenizer


#tasks = ["deeper_nesting", "denser_snippets", "hidden_numbers", "hidden_variables", "longer_snippets", "mixed_ops"]
tasks = ["deeper_nesting"]
tests = []
full_checkpoint_path = []

partial = []
for task in tasks : 
    partial.append(ood_path+"/"+task+"/data/test.txt")
tmp = []
for chk in checkpoints :
    tmp.append(checkpoint_path+"/"+chk)
tests.append(  [data_path + "/test.txt"] + partial)
full_checkpoint_path.append(tmp)


for arr in full_checkpoint_path :
    for path in arr:
        print(os.path.exists(path))



for arr in tests :
    for path in arr:
        print(path)




#test_datasets_titles = ["ID","OOD_Deeper","OOD_Denser","OOD_Hidden_numbers","OOD_Hidden_variables","OOD_Longer_snippets","OOD_Mixed_ops"]
test_datasets_titles = ["ID","OOD_Deeper"]

#alpha/beta/tokenizer
output_path = "./results"
use_busy_gpus = False
#best-model.pth
#joined_path = os.path.join(some_path, a_string)

import subprocess

def gpu_free(gpu_id, threshold=60000):  # MB
    out = subprocess.getoutput(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}")
    return (int(out.strip()) < threshold) or use_busy_gpus

# tests
# full_checkpoint_path
# full_checkpoint_path_rand_enc

jobs1= []


test_datasets = list(zip(tests[0], test_datasets_titles))
check_couple = list(zip(full_checkpoint_path[0], checkpoints,["jeff"]*len(full_checkpoint_path[0])))
jobs1.extend([(checkpoint,check_name,percentage, dataset_pth, dataset_title) for checkpoint,check_name,percentage in check_couple for dataset_pth, dataset_title in test_datasets])


# with open("/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/output.txt", "w") as f:
#     for job in jobs1:
#         print(job, file=f)
#     for job in jobs2:
#         print(job, file=f)

# print(len(jobs1),len(jobs2))


import time
import datetime

# Ensure the output directory exists so bash can create the log files there
os.makedirs(output_path, exist_ok=True)
while jobs1:
    for gpu in gpus:
        if not jobs1:
            break
        if gpu_free(gpu):
            print(f"gpu number {gpu} is free, i'll use it")
            ckpt_path, ckpt, percentage, test_pth, test_title = jobs1.pop(0)
            print(f"evaluating {ckpt_path} on {test_title}")
            checkpoint_name = f"eval_jobs1_{ckpt}_{test_title}"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define where the log file for this specific job will be saved
            log_file = os.path.join(output_path, f"log_{checkpoint_name}_{timestamp}.txt")
            
            # Added `> \"{log_file}\" 2>&1` to pipe all stdout and stderr to the log file
            subprocess.Popen(f"screen -dmS eval_job1_{ckpt}_{test_title}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[0]}\" --cptpth \"{ckpt_path}\" --datpth \"{data_path}/\" --tstpth \"{test_pth}\" --outpth \"{os.path.join(output_path, checkpoint_name)}\" > \"{log_file}\" 2>&1; exit'", shell=True)
    time.sleep(30)  # poll every 30s