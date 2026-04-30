#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=resnet_vs_blt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# BLT-VS vs pretrained ResNet — full pipeline
#
# Step 1: Extract ResNet features + compute RDMs
# Step 2: Second-order RDM comparison with BLT-VS model
# ============================================================

# --- EDIT THESE ---
MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019"
RESNET_VARIANT="resnet50"
METRIC="cosine"
RDM_TYPE="ranked"

# Paths
CSV_PATH="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
IMAGE_ROOT="/share/klab/datasets/THINGS_drift/stimuli"
ANN_RDM_DIR="analysis_outputs/ann_rdms"

# Derived paths
RESNET_RDM_DIR="analysis_outputs/resnet_rdms"
RESNET_RDM_PATH="${RESNET_RDM_DIR}/${RESNET_VARIANT}_rdms.npz"
ANN_RDM_PATH="${ANN_RDM_DIR}/${MODEL_NAME}_ann_rdms.npz"
SAVE_DIR="analysis_outputs/second_order_ann_vs_resnet"

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

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo "====================================="
echo "Step 1: Extract ResNet features + RDMs"
echo "  Variant:    $RESNET_VARIANT"
echo "  CSV:        $CSV_PATH"
echo "  Image root: $IMAGE_ROOT"
echo "  Start time: $(date)"
echo "====================================="

python analysis/rdm_generation/extract_resnet_features_and_rdms.py \
    --csv_path "$CSV_PATH" \
    --image_root "$IMAGE_ROOT" \
    --resnet_variant "$RESNET_VARIANT" \
    --metric "$METRIC" \
    --save_dir "$RESNET_RDM_DIR" \
    --batch_size 64

echo "====================================="
echo "Step 2: Second-order RDM comparison"
echo "  BLT-VS RDMs: $ANN_RDM_PATH"
echo "  ResNet RDMs:  $RESNET_RDM_PATH"
echo "  Start time:   $(date)"
echo "====================================="

python analysis/rdm_generation/second_order_rdms_ann_vs_resnet.py \
    --ann_rdm_path "$ANN_RDM_PATH" \
    --resnet_rdm_path "$RESNET_RDM_PATH" \
    --save_dir "$SAVE_DIR" \
    --metric "$METRIC" \
    --rdm_type "$RDM_TYPE" \
    --plot_panels 1

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
