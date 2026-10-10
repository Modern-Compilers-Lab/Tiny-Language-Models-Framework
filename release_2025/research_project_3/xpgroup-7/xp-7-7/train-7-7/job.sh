#!/bin/bash
#SBATCH -q c2 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --time=12:00:00
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate tinylm
#Execute the code
python sft.py