#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=2ndrdm_ann
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523"
PARENT_DIR="analysis_outputs/ann_rdms/${MODEL_NAME}"
SAVE_ROOT="analysis_outputs/second_order_ann/${MODEL_NAME}"

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

echo "====================================="
echo "Second-order RDMs (ANN time-time)"
echo "Parent dir: $PARENT_DIR"
echo "Save root:  $SAVE_ROOT"
echo "Start time: $(date)"
echo "====================================="

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

python analysis/rdm_generation/second_order_rdms_extended.py \
    --parent_dir "$PARENT_DIR" \
    --save_root "$SAVE_ROOT"

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
