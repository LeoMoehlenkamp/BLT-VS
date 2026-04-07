#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --job-name=BLT_resume
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

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
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU node/module setup is broken."
    exit 1
fi

if ! nvidia-smi; then
    echo "ERROR: nvidia-smi failed. No usable NVIDIA GPU/driver in this job."
    exit 1
fi

echo "Running PyTorch CUDA sanity check..."
python - <<'PY'
import sys
import torch

print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: PyTorch cannot see a CUDA GPU in this SLURM job.")
    sys.exit(1)

print("GPU[0]:", torch.cuda.get_device_name(0))
PY

RUN_NAME="blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451"

echo "Resuming training for run: $RUN_NAME"
echo "Starting: $(date)"

python blt_vs_model/training_code/resume_training.py \
    --run_name "$RUN_NAME" \
    --checkpoint best \
    --n_epochs 20

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"
