#!/bin/bash
#SBATCH --partition=klab-l40s
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=annrdm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

FEATURES_PATH="analysis_outputs/ann_features/blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-4__20260321_203121_features.npz"
MONKEY_PROCESSED_PATH="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz"
SAVE_DIR="analysis_outputs/ann_rdms"
PLOT_PANELS=1

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
nvidia-smi

echo "Starting ANN RDM generation..."
echo "Features: $FEATURES_PATH"
echo "Monkey processed: $MONKEY_PROCESSED_PATH"
echo "Save dir: $SAVE_DIR"
echo "Start time: $(date)"

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

python analysis/rdm_generation/generate_ann_rdms.py \
    --features_path "$FEATURES_PATH" \
    --monkey_processed_path "$MONKEY_PROCESSED_PATH" \
    --save_dir "$SAVE_DIR" \
    --plot_panels "$PLOT_PANELS"

echo "-------------------------------------"
echo "Finished: $(date)"
echo "-------------------------------------"