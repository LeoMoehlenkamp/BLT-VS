#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-7
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=resnet_rdms
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================
# CONFIGURE HERE
# =============================================
RESNET_VARIANT="resnet50"
BATCH_SIZE=32
PCA_COMPONENTS=0
PLOT_PANELS=1

# =============================================
# PATHS (usually no changes needed)
# =============================================
CSV_PATH="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
IMAGE_ROOT="/share/klab/datasets/THINGS_drift/stimuli"

MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"

FEATURES_DIR="analysis_outputs/resnet_rdms"
FEATURES_PATH="${FEATURES_DIR}/${RESNET_VARIANT}_rdms.npz"

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
echo "# RESNET RDM PIPELINE"
echo "# Variant: $RESNET_VARIANT"
echo "# Start: $(date)"
echo "######################################################################"

# =============================================
# STEP 1: Extract ResNet features (once)
# =============================================
echo ""
echo "===== STEP 1/2: Extract ResNet features ====="

if [ -f "$FEATURES_PATH" ]; then
    echo "Features already exist → skipping"
else
    python analysis/rdm_generation/extract_resnet_features_and_rdms.py \
        --csv_path "$CSV_PATH" \
        --image_root "$IMAGE_ROOT" \
        --resnet_variant "$RESNET_VARIANT" \
        --batch_size "$BATCH_SIZE" \
        --metric "cosine" \
        --pca_components "$PCA_COMPONENTS" \
        --save_dir "$FEATURES_DIR"

    if [ ! -f "$FEATURES_PATH" ]; then
        echo "ERROR: Feature file was not created: $FEATURES_PATH"
        exit 1
    fi
fi

# =============================================
# STEP 2: Generate ResNet RDMs + panels (all metric × rdm_type combos)
# =============================================
echo ""
echo "===== STEP 2/2: Generate ResNet RDMs (all metric × rdm_type combos) ====="

for METRIC in "${METRICS[@]}"; do
  for RDM_TYPE in "${RDM_TYPES[@]}"; do

    SAVE_DIR="analysis_outputs/resnet_rdms/${RESNET_VARIANT}_${METRIC}_${RDM_TYPE}"

    echo ""
    echo "  → $METRIC + $RDM_TYPE → $SAVE_DIR"

    python analysis/rdm_generation/save_resnet_rdms_extended.py \
        --features_path "$FEATURES_PATH" \
        --monkey_processed_path "$MONKEY_PROCESSED_PATH" \
        --save_dir "$SAVE_DIR" \
        --metric "$METRIC" \
        --rdm_type "$RDM_TYPE" \
        --plot_panels "$PLOT_PANELS"

  done
done

# =============================================
# SUMMARY — output locations
# =============================================
echo ""
echo "######################################################################"
echo "# PIPELINE COMPLETE: $(date)"
echo "# Variant: $RESNET_VARIANT"
echo "#"
echo "# OUTPUT LOCATIONS:"
echo "#   Features + base RDMs:  $FEATURES_PATH"
echo "#   RDM panels (per combo): analysis_outputs/resnet_rdms/${RESNET_VARIANT}_*"
echo "#"
echo "# TO COPY EVERYTHING:"
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/analysis_outputs/resnet_rdms/${RESNET_VARIANT}_* ."
echo "######################################################################"
