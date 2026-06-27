import os
import subprocess
import time
import datetime

#alpha/beta/tokenizer
#data_dir_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/gamma_1_X_gen/data_200/"
#on every architecture

#on every architecture
gpus = [0,1,2,3]
data_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/alpha_0.X/20m"
checkpoint_path = "/data/ia2921/Tiny_language_model_framework/2Training/experiment_reboot/exp4_alpha/exp4_alpha_RoPE/checkpoints1"
ood_path = "/data/ia2921/Tiny_language_model_framework/1Datasets/experiment_reboot/alpha_0.X/OOD"
evaluation_script_path = ["/data/ia2921/Tiny_language_model_framework/3Evaluations/experiment_reboot/exp4_alpha/ROPE_CL/optimus_eval.py"]

checkpoints = ["best-model.pth"]

tasks = ["deeper_nesting", "denser_snippets", "hidden_numbers", "hidden_variables", "longer_snippets", "mixed_ops"]
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

test_datasets_titles = ["ID","OOD_Deeper","OOD_Denser","OOD_Hidden_numbers","OOD_Hidden_variables","OOD_Longer_snippets","OOD_Mixed_ops"]

output_path = "./results"
# Ensure the output directory exists so log files don't fail to create
os.makedirs(output_path, exist_ok=True) 

use_busy_gpus = False

def gpu_free(gpu_id, threshold=10000):  # MB
    out = subprocess.getoutput(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}")
    return (int(out.strip()) < threshold) or use_busy_gpus

jobs1= []

test_datasets = list(zip(tests[0], test_datasets_titles))
check_couple = list(zip(full_checkpoint_path[0], checkpoints,["jeff"]*len(full_checkpoint_path[0])))
jobs1.extend([(checkpoint,check_name,percentage, dataset_pth, dataset_title) for checkpoint,check_name,percentage in check_couple for dataset_pth, dataset_title in test_datasets])

while jobs1:
    for gpu in gpus:
        if not jobs1:
            break
        if gpu_free(gpu):
            print(f"gpu number {gpu} is free, i'll use it")
            ckpt_path,ckpt,percentage, test_pth, test_title = jobs1.pop(0)
            print(f"evaluating {ckpt_path} on {test_title}")
            checkpoint_name = f"eval_jobs1_{ckpt}_{test_title}"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define where the log file will be saved
            log_file = os.path.join(output_path, f"{checkpoint_name}_{timestamp}.log")
            
            # Added > \"{log_file}\" 2>&1 before the exit command to route output/errors
            subprocess.Popen(f"screen -dmS eval_job1_{ckpt}_{test_title}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[0]}\" --cptpth \"{ckpt_path}\" --datpth \"{data_path}/\" --tstpth \"{test_pth}\" --outpth \"{os.path.join(output_path, checkpoint_name)}\" > \"{log_file}\" 2>&1; exit'", shell=True)
    time.sleep(30)  # poll every 30s