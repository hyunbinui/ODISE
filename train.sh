#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2             # num process per node
#SBATCH --cpus-per-task=1               # num cpu cores per process (total num core of a node / total num gpus of a node * requested num gpu)
#SBATCH --wait-all-nodes=1
#SBATCH --mem=50000MB                   # Using 10GB CPU Memory (MIN_MEMORY)
#SBATCH --job-name=odise
#SBATCH --partition=laal
#SBATCH --exclude=b[18,14,28]
#SBATCH --output=./slurm_logs/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate odise

export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export DETECTRON2_DATASETS="/shared/s2/lab01/dataset/zeroseg/"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

srun --cpu-bind=v --accel-bind=gn python -m tools.train_net --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 4 --amp