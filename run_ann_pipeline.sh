#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=annpipe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800"
USE_BEST=1
BATCH_SIZE=32

MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"
FEATURES_PATH="analysis_outputs/ann_features/${MODEL_NAME}_features.npz"
SAVE_DIR="analysis_outputs/ann_rdms"
PLOT_PANELS=1

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

echo "====================================="
echo "Starting ANN pipeline"
echo "Model: $MODEL_NAME"
echo "Start time: $(date)"
echo "====================================="

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo ""
echo "Step 1/2: Extract ANN features"
python analysis/rdm_generation/extract_ann_features.py \
    --model_name "$MODEL_NAME" \
    --use_best "$USE_BEST" \
    --batch_size "$BATCH_SIZE"

if [ ! -f "$FEATURES_PATH" ]; then
    echo "Feature file was not created: $FEATURES_PATH"
    exit 1
fi

echo ""
echo "Step 2/2: Generate ANN RDMs"
python analysis/rdm_generation/save_ann_rdms.py \
    --features_path "$FEATURES_PATH" \
    --monkey_processed_path "$MONKEY_PROCESSED_PATH" \
    --save_dir "$SAVE_DIR" \
    --plot_panels "$PLOT_PANELS"

echo "====================================="
echo "Pipeline finished: $(date)"
echo "====================================="