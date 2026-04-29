#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=2ndrdm_V1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_NAME="blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019"
METRIC="cosine"
RDM_TYPE="ranked"
ANN_RDM_PATH="analysis_outputs/ann_rdms/${MODEL_NAME}_${METRIC}_${RDM_TYPE}/${MODEL_NAME}_ann_rdms.npz"
MONKEY_PKL_PATH="/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_lfp_minithings/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_1-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_lfp.pkl"
STIMULUS_CSV="/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
SAVE_DIR="analysis_outputs/second_order_ann_vs_monkey/monkeyF_lfp_V1"
PLOT_PANELS=1
T_STEP=10

spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

export NCCL_SOCKET_IFNAME=lo
mkdir -p logs

source ~/startup_conda.sh
conda activate blt_vs

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version

echo "====================================="
echo "Second-order RDMs: ANN vs MonkeyF V1 (LFP)"
echo "ANN RDMs:    $ANN_RDM_PATH"
echo "Monkey PKL:  $MONKEY_PKL_PATH"
echo "Stimulus CSV: $STIMULUS_CSV"
echo "Metric:      $METRIC"
echo "RDM type:    $RDM_TYPE"
echo "Save dir:    $SAVE_DIR"
echo "Start time:  $(date)"
echo "====================================="

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

echo ""
echo "Keys in ANN RDM npz:"
python -c "
import numpy as np
d = np.load('$ANN_RDM_PATH', allow_pickle=True)
keys = sorted(d.files)
print(f'Total keys: {len(keys)}')
for k in keys[:40]:
    print(' ', k)
if len(keys) > 40:
    print(f'  ... and {len(keys)-40} more')
"
echo ""

python analysis/rdm_generation/second_order_rdms_ann_vs_monkey.py \
    --ann_rdm_path "$ANN_RDM_PATH" \
    --monkey_pkl_path "$MONKEY_PKL_PATH" \
    --stimulus_csv "$STIMULUS_CSV" \
    --save_dir "$SAVE_DIR" \
    --metric "$METRIC" \
    --rdm_type "$RDM_TYPE" \
    --plot_panels "$PLOT_PANELS" \
    --t_step "$T_STEP"

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
