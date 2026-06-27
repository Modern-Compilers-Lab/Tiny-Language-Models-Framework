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
#checkpoint_path = "/data/ia2921/Tiny_language_model_framework/2Training/gamma_200_alibi_cb/checkpoints1/"
#on every architecture
evaluation_script_path = ["/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/simple_arch.py"]
checkpoints = [
 "checkpoint_0.04.pth","checkpoint_0.08.pth","checkpoint_0.12.pth","checkpoint_0.17.pth",
 "checkpoint_0.21.pth","checkpoint_0.25.pth","checkpoint_0.29.pth","checkpoint_0.33.pth",
 "checkpoint_0.37.pth","checkpoint_0.42.pth","checkpoint_0.46.pth","checkpoint_0.50.pth",
 "checkpoint_0.54.pth","checkpoint_0.58.pth","checkpoint_0.62.pth","checkpoint_0.67.pth",
 "checkpoint_0.71.pth","checkpoint_0.75.pth","checkpoint_0.79.pth","checkpoint_0.83.pth",
 "checkpoint_0.87.pth","checkpoint_0.92.pth","checkpoint_0.96.pth","checkpoint_1.00.pth"
]

#alpha/beta/tokenizer

percentages = ["80"]
tasks = ["hidden_variables"]#["deeper_nesting", "denser_snippets", "hidden_numbers", "hidden_variables", "longer_snippets", "longer_variable_names", "mixed_ops"]
tests = []
full_checkpoint_path = []
full_checkpoint_path_rand_enc = []
for p in percentages : 
    partial = []
    for task in tasks : 
        partial.append("/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/OOD/data_"+p+"_p/"+task+"/data/test.txt")
    tmp = []
    tmp_rand_enc = []
    for chk in checkpoints :
        tmp.append("/data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train/checkpoints1_"+p+"/"+chk)
        #tmp_rand_enc.append("/data/ia2921/Tiny_language_model_framework/2Training/simple_alpha_train_random_pos/checkpoints1_"+p+"/"+chk)
    tests.append(  ["/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_80_p/test.txt"]+partial)
    full_checkpoint_path.append(tmp)
    #full_checkpoint_path_rand_enc.append(tmp_rand_enc)

# for arr in tests :
#     for path in arr:
#         print(path)
#         print(os.path.exists(path))
#     print("------------------------------------")




test_datasets_titles = ["ID","OOD_Hidden_variables"]#,"OOD_Deeper","OOD_Denser","OOD_Hidden_numbers","OOD_Hidden_variables","OOD_Longer_snippets","OOD_Longer_variables","OOD_Mixed_ops"]

#alpha/beta/tokenizer
output_path = "/data/ia2921/Tiny_language_model_framework/3Evaluations/simple_alpha/results_extended"
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
# for i in range(len(percentages)):
#     test_datasets = list(zip(tests[i], test_datasets_titles))
#     check_couple = list(zip(full_checkpoint_path_rand_enc[i], checkpoints,[percentages[i]]*len(full_checkpoint_path_rand_enc[i])))
#     jobs2.extend([(checkpoint,check_name,percentage, dataset_pth, dataset_title) for checkpoint,check_name,percentage in check_couple for dataset_pth, dataset_title in test_datasets])
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
            subprocess.Popen(f"screen -dmS eval_job1_{ckpt}_{test_title}_p_{percentage}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[0]}\" --cptpth \"{ckpt_path}\" --datpth \"/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_{percentage}_p/\" --tstpth \"{test_pth}\" --outpth \"{os.path.join(output_path, checkpoint_name)}\"; exit'", shell=True)
    time.sleep(30)  # poll every 30s


# while jobs2:
#     for gpu in gpus:
#         if not jobs2:
#             break
#         if gpu_free(gpu):
#             print(f"gpu number {gpu} is free, i'll use it")
#             ckpt_path,ckpt,percentage, test_pth, test_title = jobs2.pop(0)
#             print(f"evaluating {ckpt_path} on {test_title}")
#             checkpoint_name = f"eval_jobs2_{ckpt}_{test_title}_p_{percentage}"
#             #full_path = ckpt_path
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             subprocess.Popen(f"screen -dmS eval_job2_{ckpt}_{test_title}_p_{percentage}_{timestamp} bash -c 'source /home/ia2921/anaconda3/etc/profile.d/conda.sh && conda activate torch_env && CUDA_VISIBLE_DEVICES={gpu} python \"{evaluation_script_path[1]}\" --cptpth \"{ckpt_path}\" --datpth \"/data/ia2921/Tiny_language_model_framework/1Datasets/simple_alpha/data_{percentage}_p/\" --tstpth \"{test_pth}\" --outpth \"{os.path.join(output_path, checkpoint_name)}\"; exit'", shell=True)
#     time.sleep(30)  # poll every 30s
