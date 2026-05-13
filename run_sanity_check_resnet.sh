#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=resnet_sanity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Sanity check: pretrained ResNet50 vs pretrained ResNet101
#
# Extracts RDMs from two pretrained ResNets and correlates them
# layer-by-layer. Expect a strong diagonal pattern.
# ============================================================

RESNET_A="resnet50"
RESNET_B="resnet101"
METRIC="cosine"
RDM_TYPE="ranked"

# Paths
CSV_PATH="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
IMAGE_ROOT="/share/klab/datasets/THINGS_drift/stimuli"
RDM_DIR="analysis_outputs/resnet_rdms"
SAVE_DIR="analysis_outputs/sanity_check_resnet"

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

# Step 1: Extract RDMs for ResNet A (pretrained)
echo "====================================="
echo "Step 1a: Extract pretrained ${RESNET_A} RDMs"
echo "  Start time: $(date)"
echo "====================================="

python analysis/rdm_generation/extract_resnet_features_and_rdms.py \
    --csv_path "$CSV_PATH" \
    --image_root "$IMAGE_ROOT" \
    --resnet_variant "$RESNET_A" \
    --metric "$METRIC" \
    --save_dir "$RDM_DIR" \
    --batch_size 64

# Step 2: Extract RDMs for ResNet B (pretrained)
echo "====================================="
echo "Step 1b: Extract pretrained ${RESNET_B} RDMs"
echo "  Start time: $(date)"
echo "====================================="

python analysis/rdm_generation/extract_resnet_features_and_rdms.py \
    --csv_path "$CSV_PATH" \
    --image_root "$IMAGE_ROOT" \
    --resnet_variant "$RESNET_B" \
    --metric "$METRIC" \
    --save_dir "$RDM_DIR" \
    --batch_size 64

# Step 3: Correlate
echo "====================================="
echo "Step 2: Correlate ${RESNET_A} vs ${RESNET_B}"
echo "  Start time: $(date)"
echo "====================================="

python analysis/rdm_generation/sanity_check_resnet_vs_resnet.py \
    --rdm_path_a "${RDM_DIR}/${RESNET_A}_rdms.npz" \
    --rdm_path_b "${RDM_DIR}/${RESNET_B}_rdms.npz" \
    --metric "$METRIC" \
    --rdm_type "$RDM_TYPE" \
    --save_dir "$SAVE_DIR"

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
