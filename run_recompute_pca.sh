#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=recompute_pca
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================
# CONFIGURE HERE
# =============================================
MODEL_NAME="blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143"
USE_BEST=1

# =============================================
# ENVIRONMENT SETUP
# =============================================
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

export PYTHONPATH="/share/klab/danthes/lemoehlenkam/BLT-VS/blt_vs_model/training_code:${PYTHONPATH}"

echo ""
echo "######################################################################"
echo "# RECOMPUTE PCA"
echo "# Model:    $MODEL_NAME"
echo "# use_best: $USE_BEST"
echo "# Start:    $(date)"
echo "######################################################################"

python blt_vs_model/training_code/recompute_pca.py \
    --model_name "$MODEL_NAME" \
    --use_best "$USE_BEST"

echo ""
echo "-------------------------------------"
echo "PCA plots saved to:"
echo "  logs/perf_logs/${MODEL_NAME}/"
echo "Finished: $(date)"
echo "-------------------------------------"
