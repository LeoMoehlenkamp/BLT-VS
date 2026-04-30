#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=resnet_vs_monkey
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Monkey vs pretrained ResNet — second-order RDM comparison
#
# Same pipeline as second_order_rdms_ann_vs_monkey.py but with
# ResNet layers instead of BLT-VS areas+timesteps.
# ============================================================

# --- EDIT THESE ---
RESNET_VARIANT="resnet50"
METRIC="cosine"
RDM_TYPE="ranked"

# Paths
MONKEY_PKL="/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_lfp_minithings/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_lfp.pkl"
STIMULUS_CSV="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"

# Derived paths
RESNET_RDM_DIR="analysis_outputs/resnet_rdms"
RESNET_RDM_PATH="${RESNET_RDM_DIR}/${RESNET_VARIANT}_rdms.npz"
SAVE_DIR="analysis_outputs/second_order_resnet_vs_monkey"

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
echo "Monkey vs ResNet second-order RDMs"
echo "  ResNet RDMs:  $RESNET_RDM_PATH"
echo "  Monkey PKL:   $MONKEY_PKL"
echo "  Metric:       $METRIC"
echo "  RDM type:     $RDM_TYPE"
echo "  Start time:   $(date)"
echo "====================================="

python analysis/rdm_generation/second_order_rdms_resnet_vs_monkey.py \
    --resnet_rdm_path "$RESNET_RDM_PATH" \
    --monkey_pkl_path "$MONKEY_PKL" \
    --stimulus_csv "$STIMULUS_CSV" \
    --save_dir "$SAVE_DIR" \
    --metric "$METRIC" \
    --rdm_type "$RDM_TYPE" \
    --plot_panels 1

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
