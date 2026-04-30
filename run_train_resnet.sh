#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-8
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=resnet50_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Train ResNet50 on the SAME data / transforms as BLT-VS
# Uses the existing train_net_copy_hooks.py with --network rn50
# ============================================================

DATASET="miniecoset"          # ecoset | miniecoset | imagenet
DATASET_MODE=0                # 0 = EcoSet, 1 = FakeData, 2 = CIFAR100
EPOCHS=60
BATCH_SIZE=256
LR=7.5e-4
NUM_WORKERS=8
RUN_NAME="resnet50__miniecoset"   # leave empty for auto-generated name

# --- Environment setup ---
spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

source ~/startup_conda.sh
conda activate blt_vs
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo "====================================="
echo "Training ResNet50"
echo "  Dataset:    $DATASET (mode $DATASET_MODE)"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Start time: $(date)"
echo "====================================="

python blt_vs_model/training_code/train_net_copy_hooks.py \
    --network rn50 \
    --name "$RUN_NAME" \
    --dataset_mode "$DATASET_MODE" \
    --dataset "$DATASET" \
    --timesteps 1 \
    --lateral_connections 0 \
    --topdown_connections 0 \
    --skip_connections 0 \
    --bio_unroll 0 \
    --batch_size "$BATCH_SIZE" \
    --batch_size_val_test "$BATCH_SIZE" \
    --n_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --num_workers "$NUM_WORKERS" \
    --grad_clipping 1

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
