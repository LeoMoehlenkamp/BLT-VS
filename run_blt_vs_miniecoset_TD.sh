#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

python blt_vs_model/training_code/train_net_copy_hooks.py \
    --network blt_vs_bottleneck \
    --bottlenecks "V1->V2:64,V2->V3:64,V3->V4:64,V4->LOC:64,V1->LGN_td:64,V2->V1_td:64,V3->V2_td:64,V4->V3_td:64,LOC->V4_td:64" \
    --name "blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-TD" \
    --dataset_mode 0 \
    --dataset miniecoset \
    --timesteps 12 \
    --lateral_connections 1 \
    --topdown_connections 1 \
    --skip_connections 0 \
    --bio_unroll 1 \
    --batch_size 64 \
    --batch_size_val_test 64 \
    --n_epochs 60 \
    --learning_rate 7.5e-4 \
    --num_workers 4

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"