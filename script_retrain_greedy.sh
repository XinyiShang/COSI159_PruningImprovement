#!/bin/bash

#SBATCH --job-name=retain_greedy
#SBATCH --output=output_retrain_greedy.txt
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:TitanX:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xinyishang@brandeis.edu

# Load modules required for your job
module load cuda/11.7

# Activate the Anaconda environment that contains PyTorch
source /home/xinyishang/ENTER/bin/activate /home/xinyishang/ENTER


# Run your Python code
srun python retrain_greedy.py