#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=full_pipe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================
# CONFIGURE HERE
# =============================================
MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bnall96_BU-TD-Skip__20260423_085848"
USE_BEST=1
BATCH_SIZE=32
PLOT_PANELS=1
T_STEP=10

# =============================================
# PATHS (usually no changes needed)
# =============================================
MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"
FEATURES_PATH="analysis_outputs/ann_features/${MODEL_NAME}_features.npz"
LOSS_NPZ_PATH="logs/perf_logs/${MODEL_NAME}/loss_${MODEL_NAME}.npz"

METRICS=("cosine" "euclidean")
RDM_TYPES=("ranked" "raw")

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

echo ""
echo "######################################################################"
echo "# FULL ANALYSIS PIPELINE"
echo "# Model: $MODEL_NAME"
echo "# Start: $(date)"
echo "######################################################################"

# =============================================
# STEP 1: Extract ANN features (once)
# =============================================
echo ""
echo "===== STEP 1/5: Extract ANN features ====="

if [ -f "$FEATURES_PATH" ]; then
    echo "Features already exist → skipping"
else
    python analysis/rdm_generation/extract_ann_features.py \
        --model_name "$MODEL_NAME" \
        --use_best "$USE_BEST" \
        --batch_size "$BATCH_SIZE"

    if [ ! -f "$FEATURES_PATH" ]; then
        echo "ERROR: Feature file was not created: $FEATURES_PATH"
        exit 1
    fi
fi

# =============================================
# STEP 2: Generate ANN RDMs (all combinations)
# =============================================
echo ""
echo "===== STEP 2/5: Generate ANN RDMs (all metric × rdm_type combos) ====="

for METRIC in "${METRICS[@]}"; do
  for RDM_TYPE in "${RDM_TYPES[@]}"; do

    SAVE_DIR="analysis_outputs/ann_rdms/${MODEL_NAME}_${METRIC}_${RDM_TYPE}"

    echo ""
    echo "  → $METRIC + $RDM_TYPE → $SAVE_DIR"

    python analysis/rdm_generation/save_ann_rdms_extended.py \
        --features_path "$FEATURES_PATH" \
        --monkey_processed_path "$MONKEY_PROCESSED_PATH" \
        --save_dir "$SAVE_DIR" \
        --plot_panels "$PLOT_PANELS" \
        --metric "$METRIC" \
        --rdm_type "$RDM_TYPE"

  done
done

# =============================================
# STEP 3: Second-order RDMs — ANN time-time
# =============================================
echo ""
echo "===== STEP 3/5: Second-order RDMs (ANN time-time) ====="

SECOND_ORDER_ANN_DIR="analysis_outputs/second_order_ann/${MODEL_NAME}"

python analysis/rdm_generation/second_order_rdms_extended.py \
    --parent_dir "analysis_outputs/ann_rdms" \
    --save_root "$SECOND_ORDER_ANN_DIR" \
    --model_name "$MODEL_NAME"

# =============================================
# STEP 4: Second-order RDMs — ANN vs Monkey
# =============================================
echo ""
echo "===== STEP 4/5: Second-order RDMs (ANN vs Monkey) ====="

SECOND_ORDER_MONKEY_DIR="analysis_outputs/second_order_ann_vs_monkey/${MODEL_NAME}"

for METRIC in "${METRICS[@]}"; do
  for RDM_TYPE in "${RDM_TYPES[@]}"; do

    ANN_RDM_FILE="analysis_outputs/ann_rdms/${MODEL_NAME}_${METRIC}_${RDM_TYPE}/${MODEL_NAME}_ann_rdms.npz"
    MONKEY_SAVE_DIR="${SECOND_ORDER_MONKEY_DIR}/${METRIC}_${RDM_TYPE}"

    if [ ! -f "$ANN_RDM_FILE" ]; then
        echo "  ✗ Skipping $METRIC + $RDM_TYPE (ANN RDM file not found: $ANN_RDM_FILE)"
        continue
    fi

    echo ""
    echo "  → $METRIC + $RDM_TYPE"

    python analysis/rdm_generation/second_order_rdms_ann_vs_monkey.py \
        --ann_rdm_path "$ANN_RDM_FILE" \
        --save_dir "$MONKEY_SAVE_DIR" \
        --metric "$METRIC" \
        --rdm_type "$RDM_TYPE" \
        --plot_panels "$PLOT_PANELS" \
        --t_step "$T_STEP"

  done
done

# =============================================
# STEP 5: Recurrence Score Summary
# =============================================
echo ""
echo "===== STEP 5/5: Recurrence Score Summary ====="

if [ -f "$LOSS_NPZ_PATH" ]; then
    python analysis/bn_experiments_plots/ReccurenceScoreSummarySingleRun.py \
        --npz_path "$LOSS_NPZ_PATH"
else
    echo "  ✗ Loss NPZ not found: $LOSS_NPZ_PATH → skipping recurrence score"
fi

# =============================================
# SUMMARY — output locations
# =============================================
echo ""
echo "######################################################################"
echo "# PIPELINE COMPLETE: $(date)"
echo "# Model: $MODEL_NAME"
echo "#"
echo "# OUTPUT LOCATIONS:"
echo "#   Features:                $FEATURES_PATH"
echo "#   ANN RDMs (per combo):   analysis_outputs/ann_rdms/${MODEL_NAME}_*"
echo "#   2nd-order ANN:          $SECOND_ORDER_ANN_DIR"
echo "#   2nd-order ANN vs Monkey:$SECOND_ORDER_MONKEY_DIR"
echo "#   Recurrence score:       $(dirname $LOSS_NPZ_PATH)"
echo "#"
echo "# TO COPY EVERYTHING:"
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/analysis_outputs/ann_rdms/${MODEL_NAME}_* ."
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/$SECOND_ORDER_ANN_DIR ."
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/$SECOND_ORDER_MONKEY_DIR ."
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/$(dirname $LOSS_NPZ_PATH) ."
echo "######################################################################"
