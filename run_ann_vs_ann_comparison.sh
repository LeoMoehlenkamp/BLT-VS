#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ann_vs_ann
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================
# CONFIGURE HERE
# =============================================
MODEL_A="blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143"
MODEL_B="blt_vs_pretrained_ecoset"

LABEL_A="trained"
LABEL_B="pretrained"

METRICS=("cosine" "euclidean")
RDM_TYPES=("ranked" "raw")

# =============================================
# ENVIRONMENT SETUP
# =============================================
spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"

source ~/startup_conda.sh
conda activate blt_vs
export PYTHONWARNINGS="ignore::FutureWarning"

mkdir -p logs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo ""
echo "######################################################################"
echo "# ANN vs ANN COMPARISON"
echo "# Model A: $MODEL_A ($LABEL_A)"
echo "# Model B: $MODEL_B ($LABEL_B)"
echo "# Start: $(date)"
echo "######################################################################"

SAVE_ROOT="analysis_outputs/second_order_ann_vs_ann/${LABEL_A}_vs_${LABEL_B}"

for METRIC in "${METRICS[@]}"; do
  for RDM_TYPE in "${RDM_TYPES[@]}"; do

    RDM_PATH_A="analysis_outputs/ann_rdms/${MODEL_A}_${METRIC}_${RDM_TYPE}/${MODEL_A}_ann_rdms.npz"
    RDM_PATH_B="analysis_outputs/ann_rdms/${MODEL_B}_${METRIC}_${RDM_TYPE}/${MODEL_B}_ann_rdms.npz"

    SAVE_DIR="${SAVE_ROOT}/${METRIC}_${RDM_TYPE}"

    if [ ! -f "$RDM_PATH_A" ]; then
        echo "  ✗ Skipping $METRIC + $RDM_TYPE (Model A RDM not found: $RDM_PATH_A)"
        continue
    fi
    if [ ! -f "$RDM_PATH_B" ]; then
        echo "  ✗ Skipping $METRIC + $RDM_TYPE (Model B RDM not found: $RDM_PATH_B)"
        continue
    fi

    echo ""
    echo "===== $METRIC + $RDM_TYPE ====="

    python analysis/rdm_generation/second_order_rdms_ann_vs_ann.py \
        --rdm_path_a "$RDM_PATH_A" \
        --rdm_path_b "$RDM_PATH_B" \
        --save_dir "$SAVE_DIR" \
        --metric "$METRIC" \
        --rdm_type "$RDM_TYPE" \
        --label_a "$LABEL_A" \
        --label_b "$LABEL_B"

  done
done

echo ""
echo "######################################################################"
echo "# DONE: $(date)"
echo "# Outputs: $SAVE_ROOT"
echo "#"
echo "# TO COPY:"
echo "#   scp -r lemoehlenkam@hpc3.rz.uos.de:/share/klab/danthes/lemoehlenkam/BLT-VS/$SAVE_ROOT ."
echo "######################################################################"
