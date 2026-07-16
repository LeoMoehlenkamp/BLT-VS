#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=unique_var
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Unique-variance analysis: BLT-VS vs ResNet → Monkey
#
# How much variance in monkey neural RDMs is uniquely explained
# by BLT-VS vs by ResNet?
# ============================================================

# ===== EDIT THESE =====

# BLT-VS model RDMs (.npz from save_ann_rdms_extended.py)
ANN_RDM_PATH="analysis_outputs/ann_rdms/blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms.npz"

# ResNet RDMs — choose one mode:
#   Option A: TIMM pkl
# RESNET_RDM_PKL="/share/klab/danthes/danthes/THINGS_Drift/datasets/TIMM/resnet18/rdms-resnet18-metric_cosine-normalization_None-pca_1000.pkl"
# RESNET_RDM_NPZ=""
#   Option B: your own extraction npz
RESNET_RDM_PKL=""
RESNET_RDM_NPZ="analysis_outputs/resnet_rdms/resnet50_rdms.npz"

# Monkey data
MONKEY_PKL="/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_lfp_minithings/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_lfp.pkl"
STIMULUS_CSV="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"

# RDM settings
METRIC="cosine"
RDM_TYPE="ranked"

# Monkey time window
T_START=0
T_END=400
T_STEP=10

# Layer selection: "best" or "fixed"
LAYER_SELECTION="best"

# BLT-VS area filter (comma-separated, or leave empty for all areas)
# Examples: "V4" or "V1,V4,LOC" or ""
BLT_AREA=""

# Only needed if LAYER_SELECTION="fixed":
# BLT_TIMESTEP=8
# RESNET_LAYER="layer4.2.conv3"

# Regression settings
ZSCORE=1
POSITIVE=1

SAVE_DIR="analysis_outputs/unique_variance"

# ===== END EDIT =====

# --- Environment setup ---
spack load miniconda3
spack load git
eval "$(conda shell.bash hook)"

source ~/startup_conda.sh
conda activate blt_vs
export PYTHONWARNINGS="ignore::FutureWarning"

mkdir -p logs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

# --- Build command ---
CMD="python analysis/rdm_generation/unique_variance_blt_vs_resnet.py"
CMD+=" --ann_rdm_path $ANN_RDM_PATH"
CMD+=" --monkey_pkl_path $MONKEY_PKL"
CMD+=" --stimulus_csv $STIMULUS_CSV"
CMD+=" --metric $METRIC"
CMD+=" --rdm_type $RDM_TYPE"
CMD+=" --t_start $T_START"
CMD+=" --t_end $T_END"
CMD+=" --t_step $T_STEP"
CMD+=" --layer_selection $LAYER_SELECTION"
CMD+=" --zscore $ZSCORE"
CMD+=" --positive $POSITIVE"
CMD+=" --save_dir $SAVE_DIR"

# ResNet source
if [ -n "$RESNET_RDM_PKL" ]; then
    CMD+=" --resnet_rdm_pkl $RESNET_RDM_PKL"
elif [ -n "$RESNET_RDM_NPZ" ]; then
    CMD+=" --resnet_rdm_npz $RESNET_RDM_NPZ"
else
    echo "ERROR: Set either RESNET_RDM_PKL or RESNET_RDM_NPZ"
    exit 1
fi

# Optional area filter
if [ -n "$BLT_AREA" ]; then
    CMD+=" --blt_area $BLT_AREA"
fi

# Fixed-layer args (only if LAYER_SELECTION=fixed)
if [ "$LAYER_SELECTION" = "fixed" ]; then
    CMD+=" --blt_timestep ${BLT_TIMESTEP:-8}"
    CMD+=" --resnet_layer ${RESNET_LAYER:-layer4.2.conv3}"
fi

echo "====================================="
echo "Unique Variance: BLT-VS vs ResNet → Monkey"
echo "  ANN RDMs:      $ANN_RDM_PATH"
echo "  ResNet:         ${RESNET_RDM_PKL:-$RESNET_RDM_NPZ}"
echo "  Layer select:   $LAYER_SELECTION"
echo "  BLT area:       ${BLT_AREA:-all}"
echo "  Metric:         $METRIC / $RDM_TYPE"
echo "  Time window:    ${T_START}-${T_END} step ${T_STEP}"
echo "  Start time:     $(date)"
echo "====================================="

eval $CMD

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
