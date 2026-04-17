#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=BLT_resume
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
echo "Node: $(hostname)"
which python
python --version
nvidia-smi

echo "Starting resume training..."

python blt_vs_model/training_code/resume_training.py \
    --run_name "blt_vs_bottleneck__miniecoset__ts12__bn-bnall64skip__20260416_130242" \
    --checkpoint best \
    --n_epochs 22 \
    --learning_rate 7.5e-5

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"
