#!/bin/bash
#SBATCH --job-name=prepare_fid_stats
#SBATCH --output=logs/prepare_fid_stats.out
#SBATCH --error=logs/prepare_fid_stats.err
#SBATCH --time=1:01:00
#SBATCH --account=kolemen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

source .venv/bin/activate
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=4 prepare_fid_stats.py --dataset=imagenet --img_size=128