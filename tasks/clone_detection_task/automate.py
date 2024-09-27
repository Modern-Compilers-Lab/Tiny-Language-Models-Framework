import subprocess
import sys
import os

def run_script(script_name, args):

    subprocess.run(['python', script_name] + args, check=True)

def automate_process(programs_num):
    level_folder = f"all_levels"
    for i in range(1,5):
        dataset_file = f"dataset_level{i}.txt"
        outputs_json = f"outputs_level{i}.json"
        clones_json = f"clones_level{i}.json"
        clone_pairs_txt = f"code_snippets_pairs.txt"
        
        if i != 4:
            run_script('./tasks/clone_detection_task/generating_data_process/code_generator.py', ['--num_programs', programs_num, f'--level', f'{i}.2', '--filename',     dataset_file])
        else:
            run_script('./tasks/clone_detection_task/generating_data_process/code_generator.py', ['--num_programs', programs_num, f'--level', f'{i}.1', '--filename', dataset_file])
    
        run_script('./tasks/clone_detection_task/generating_data_process/code_execution.py', [f'--level', f'{i}',f'--dataset-file', dataset_file, '--output-file',     outputs_json])

        run_script('./tasks/clone_detection_task/generating_data_process/identify_clones.py', ['--input-file', outputs_json, '--output-file', clones_json])

        run_script('./tasks/clone_detection_task/generating_data_process/generating_binaries.py', ['--input-clones',clones_json,'--input-snippets', dataset_file, '--output-file', clone_pairs_txt])
        
        os.remove(dataset_file)
        os.remove(outputs_json)
        os.remove(clones_json)

    print(f"Process for all levels completed successfully!")

if __name__ == "__main__":

    programs_num = sys.argv[2]
    automate_process(programs_num)
