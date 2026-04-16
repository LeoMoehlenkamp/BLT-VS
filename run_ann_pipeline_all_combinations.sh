#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=annpipe_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523"
USE_BEST=1
BATCH_SIZE=32

MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"
FEATURES_PATH="analysis_outputs/ann_features/${MODEL_NAME}_features.npz"
PLOT_PANELS=1

METRICS=("cosine" "euclidean")
RDM_TYPES=("ranked" "raw")

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

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo "====================================="
echo "Starting ANN pipeline MULTI"
echo "Model: $MODEL_NAME"
echo "====================================="

# 🔥 STEP 1: Features nur EINMAL berechnen
if [ ! -f "$FEATURES_PATH" ]; then
    echo "Extracting features..."
    python analysis/rdm_generation/extract_ann_features.py \
        --model_name "$MODEL_NAME" \
        --use_best "$USE_BEST" \
        --batch_size "$BATCH_SIZE"
else
    echo "Features already exist → skipping"
fi

# 🔥 STEP 2: Alle Kombinationen durchgehen
for METRIC in "${METRICS[@]}"; do
  for RDM_TYPE in "${RDM_TYPES[@]}"; do

    SAVE_DIR="analysis_outputs/ann_rdms/${MODEL_NAME}_${METRIC}_${RDM_TYPE}"

    echo ""
    echo "Running: $METRIC + $RDM_TYPE"
    echo "Saving to: $SAVE_DIR"

    python analysis/rdm_generation/save_ann_rdms_extended.py \
        --features_path "$FEATURES_PATH" \
        --monkey_processed_path "$MONKEY_PROCESSED_PATH" \
        --save_dir "$SAVE_DIR" \
        --plot_panels "$PLOT_PANELS" \
        --metric "$METRIC" \
        --rdm_type "$RDM_TYPE"

  done
done

echo "====================================="
echo "ALL DONE: $(date)"
echo "====================================="