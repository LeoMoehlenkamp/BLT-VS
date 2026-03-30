#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=annfeat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-4__20260321_203121"
USE_BEST=1
BATCH_SIZE=32

spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

source ~/startup_conda.sh
conda activate blt_vs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

echo "Starting ANN feature extraction..."
echo "Model: $MODEL_NAME"
echo "Start time: $(date)"

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

python analysis/rdm_generation/extract_ann_features.py \
    --model_name "$MODEL_NAME" \
    --use_best "$USE_BEST" \
    --batch_size "$BATCH_SIZE"

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"