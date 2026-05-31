#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=BNall16
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
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo "Starting training..."

# =============================================
# TRAINING
# =============================================
TRAIN_LOG=$(mktemp)

python blt_vs_model/training_code/train_net_copy_hooks.py \
    --network blt_vs_bottleneck \
    --bottlenecks "V1->V2:16,V2->V3:16,V3->V4:16,V4->LOC:16,V1->LGN_td:16,V2->V1_td:16,V3->V2_td:16,V4->V3_td:16,LOC->V4_td:16,V1->V4_skip:16,V4->V1_skip:16" \
    --name "blt_vs_bottleneck__miniecoset__ts12__bnall16_BU-TD-Skip" \
    --dataset_mode 0 \
    --dataset miniecoset \
    --timesteps 12 \
    --lateral_connections 1 \
    --topdown_connections 1 \
    --skip_connections 1 \
    --bio_unroll 1 \
    --batch_size 64 \
    --batch_size_val_test 64 \
    --n_epochs 60 \
    --learning_rate 7.5e-4 \
    --num_workers 4 \
    2>&1 | tee "$TRAIN_LOG"

echo "-------------------------------------"
echo "Training finished: $(date)"
echo "-------------------------------------"

# =============================================
# EXTRACT MODEL NAME FROM TRAINING OUTPUT
# =============================================
MODEL_NAME=$(grep "Log_folders:" "$TRAIN_LOG" | sed 's|.*logs/perf_logs/||' | sed 's| .*||')
rm -f "$TRAIN_LOG"

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: Could not extract model name from training output."
    echo "Skipping analysis pipeline."
    exit 1
fi

echo ""
echo "Detected model name: $MODEL_NAME"

# =============================================
# FULL ANALYSIS PIPELINE
# =============================================
USE_BEST=1
BATCH_SIZE=32
PLOT_PANELS=1
T_STEP=10

MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"
FEATURES_PATH="analysis_outputs/ann_features/${MODEL_NAME}_features.npz"
LOSS_NPZ_PATH="logs/perf_logs/${MODEL_NAME}/loss_${MODEL_NAME}.npz"

METRICS=("cosine" "euclidean")
RDM_TYPES=("ranked" "raw")

echo ""
echo "######################################################################"
echo "# FULL ANALYSIS PIPELINE"
echo "# Model: $MODEL_NAME"
echo "# Start: $(date)"
echo "######################################################################"

# --- STEP 1: Extract ANN features ---
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

# --- STEP 2: Generate ANN RDMs ---
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

# --- STEP 3: Second-order RDMs — ANN time-time ---
echo ""
echo "===== STEP 3/5: Second-order RDMs (ANN time-time) ====="

SECOND_ORDER_ANN_DIR="analysis_outputs/second_order_ann/${MODEL_NAME}"

python analysis/rdm_generation/second_order_rdms_extended.py \
    --parent_dir "analysis_outputs/ann_rdms" \
    --save_root "$SECOND_ORDER_ANN_DIR" \
    --model_name "$MODEL_NAME"

# --- STEP 4: Second-order RDMs — ANN vs Monkey ---
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

# --- STEP 5: Recurrence Score Summary ---
echo ""
echo "===== STEP 5/5: Recurrence Score Summary ====="

if [ -f "$LOSS_NPZ_PATH" ]; then
    python analysis/bn_experiments_plots/ReccurenceScoreSummarySingleRun.py \
        --npz_path "$LOSS_NPZ_PATH"
else
    echo "  ✗ Loss NPZ not found: $LOSS_NPZ_PATH → skipping recurrence score"
fi

# =============================================
# SUMMARY + COPY COMMANDS
# =============================================
HPC_BASE="lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS"

echo ""
echo "######################################################################"
echo "# PIPELINE COMPLETE: $(date)"
echo "# Model: $MODEL_NAME"
echo "#"
echo "# OUTPUT LOCATIONS:"
echo "#   Training logs:          logs/perf_logs/${MODEL_NAME}/"
echo "#   Network weights:        logs/net_params/${MODEL_NAME}/"
echo "#   Features:               $FEATURES_PATH"
echo "#   ANN RDMs (per combo):   analysis_outputs/ann_rdms/${MODEL_NAME}_*"
echo "#   2nd-order ANN:          $SECOND_ORDER_ANN_DIR"
echo "#   2nd-order ANN vs Monkey:$SECOND_ORDER_MONKEY_DIR"
echo "#"
echo "# COPY COMMANDS (run on your local machine):"
echo "#"
echo "# Training logs + plots:"
echo "scp -r ${HPC_BASE}/logs/perf_logs/${MODEL_NAME} ."
echo "#"
echo "# Network weights:"
echo "scp -r ${HPC_BASE}/logs/net_params/${MODEL_NAME} ."
echo "#"
echo "# Features:"
echo "scp ${HPC_BASE}/${FEATURES_PATH} ."
echo "#"
echo "# ANN RDMs (all combos):"
echo "scp -r ${HPC_BASE}/analysis_outputs/ann_rdms/${MODEL_NAME}_* ."
echo "#"
echo "# 2nd-order ANN:"
echo "scp -r ${HPC_BASE}/${SECOND_ORDER_ANN_DIR} ."
echo "#"
echo "# 2nd-order ANN vs Monkey:"
echo "scp -r ${HPC_BASE}/${SECOND_ORDER_MONKEY_DIR} ."
echo "#"
echo "# ALL AT ONCE (one command):"
echo "mkdir -p ${MODEL_NAME}_results && cd ${MODEL_NAME}_results && scp -r ${HPC_BASE}/logs/perf_logs/${MODEL_NAME} ./perf_logs/ && scp -r ${HPC_BASE}/logs/net_params/${MODEL_NAME} ./net_params/ && scp ${HPC_BASE}/${FEATURES_PATH} . && scp -r ${HPC_BASE}/analysis_outputs/ann_rdms/${MODEL_NAME}_* ./ann_rdms/ && scp -r ${HPC_BASE}/${SECOND_ORDER_ANN_DIR} ./second_order_ann/ && scp -r ${HPC_BASE}/${SECOND_ORDER_MONKEY_DIR} ./second_order_ann_vs_monkey/"
echo "######################################################################"
