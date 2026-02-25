#!/bin/bash
#SBATCH --job-name=blt_vs_miniecoset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/blt_vs_%j.out
#SBATCH --error=logs/blt_vs_%j.err
#SBATCH --requeue
#SBATCH --signal=SIGTERM@180

echo "-------------------------------------"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "-------------------------------------"

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

# Load CUDA
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8

# Activate Conda
source ~/startup_conda.sh
conda activate blt_vs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

echo "Starting training..."

python train_net.py \
    --network blt_vs_bottleneck \
    --dataset_mode 0 \
    --dataset miniecoset \
    --timesteps 10 \
    --lateral_connections 1 \
    --topdown_connections 1 \
    --skip_connections 0 \
    --bio_unroll 1 \
    --batch_size 256 \
    --batch_size_val_test 256 \
    --n_epochs 40 \
    --learning_rate 7.5e-4 \
    --num_workers 8

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"