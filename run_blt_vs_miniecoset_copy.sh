#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100.80gb
#SBATCH --time=2:00:00
#SBATCH --job-name=BLT
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

# Activate Conda
source ~/startup_conda.sh
conda activate blt_vs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

echo "Starting training..."

python blt_vs_model/training_code/train_net_copy.py \
    --network blt_vs_bottleneck \
    --dataset_mode 0 \
    --dataset miniecoset \
    --timesteps 3 \
    --lateral_connections 1 \
    --topdown_connections 1 \
    --skip_connections 0 \
    --bio_unroll 1 \
    --batch_size 256 \
    --batch_size_val_test 256 \
    --n_epochs 5 \
    --learning_rate 7.5e-4 \
    --num_workers 8

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"