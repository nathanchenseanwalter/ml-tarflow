#!/bin/bash
#SBATCH --job-name=cond_imgnet_tune_32_gpu
#SBATCH --output=logs/%j_cond_imgnet_tune_32_gpu.out
#SBATCH --error=logs/%j_cond_imgnet_tune_32_gpu.err
#SBATCH --time=24:00:00
#SBATCH --qos=gpu-medium
#SBATCH --account=pppl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29501

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --dataset=imagenet --img_size=128 --channel_size=3 \
  --patch_size=8 --channels=768 --blocks=8 --layers_per_block=8 \
  --noise_std=0.15 --batch_size=768 --epochs=320 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1 \
  --sample_freq=5 --logdir=runs/cond-imagenet128-finetune --compile --accum_steps=1 --num_workers=8 \
  --resume=models/imagenet_model_4_1024_8_8_0.15.pth