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
trap 'echo "ERROR at line $LINENO while running: $BASH_COMMAND" >&2' ERR

echo "[1/7] Loading modules..."
spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
echo "[1/7] Modules loaded."

echo "[2/7] Initializing conda shell hook..."
eval "$(conda shell.bash hook)"
echo "[2/7] Conda hook initialized."

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

echo "[3/7] Activating conda environment..."
if [ -f "$HOME/startup_conda.sh" ]; then
    set +e
    source "$HOME/startup_conda.sh"
    set -e
fi
conda activate blt_vs
echo "[3/7] Conda environment activated."

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "[4/7] Checking GPU visibility with nvidia-smi..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU node/module setup is broken."
    exit 1
fi

if ! nvidia-smi; then
    echo "ERROR: nvidia-smi failed. No usable NVIDIA GPU/driver in this job."
    exit 1
fi
echo "[4/7] nvidia-smi check passed."

echo "[5/7] Skipping strict PyTorch CUDA pre-check (matching standard training script behavior)."

RUN_NAME="blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451"

echo "Resuming training for run: $RUN_NAME"
echo "Starting: $(date)"

echo "[6/7] Launching resume training..."
python blt_vs_model/training_code/resume_training.py \
    --run_name "$RUN_NAME" \
    --checkpoint best \
    --n_epochs 20
echo "[6/7] Resume training finished."

echo "[7/7] Job completed successfully."
echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"
