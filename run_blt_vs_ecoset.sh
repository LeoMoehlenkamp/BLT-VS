#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-7
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --job-name=blt_ecoset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# Train a BLT-VS model on full EcoSet, inheriting config from
# a previous MiniEcoSet run.
#
# Usage:
#   1. Set SOURCE_RUN to the log folder of a previous run
#   2. Override any args you want to change below
#   3. sbatch run_blt_vs_ecoset.sh
#
# The script reads the args.json from the source run and
# rebuilds the exact same command line, then applies overrides.
# ============================================================

# --- SOURCE: which previous run to inherit from ---
SOURCE_RUN="logs/perf_logs/blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800"

# --- OVERRIDES: only list what you want to change ---
# These will replace the corresponding values from args.json.
# Leave empty ("") to keep the original value.
OVERRIDE_DATASET="ecoset"
OVERRIDE_NAME="blt_vs_bottleneck__ecoset__ts12__bnnone_BU"
OVERRIDE_BATCH_SIZE="64"
OVERRIDE_BATCH_SIZE_VAL_TEST="64"
OVERRIDE_N_EPOCHS="15"
OVERRIDE_LEARNING_RATE=""       # empty = keep from source
OVERRIDE_NUM_WORKERS="2"

# --- Environment setup ---
spack load miniconda3
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

export NCCL_SOCKET_IFNAME=lo
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p logs

source ~/startup_conda.sh
conda activate blt_vs
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
nvidia-smi

cd /share/klab/danthes/lemoehlenkam/BLT-VS || exit 1

# ============================================================
# Build command from source args.json + overrides
# ============================================================
ARGS_JSON="${SOURCE_RUN}/args.json"

if [ ! -f "$ARGS_JSON" ]; then
    echo "ERROR: args.json not found at $ARGS_JSON"
    echo "Available runs:"
    ls logs/perf_logs/ 2>/dev/null || echo "  (no runs found)"
    exit 1
fi

echo "====================================="
echo "Loading config from: $ARGS_JSON"
echo "====================================="

# Read all args from the source run's args.json
SRC_NETWORK=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['network'])")
SRC_TIMESTEPS=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['timesteps'])")
SRC_LATERAL=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['lateral_connections'])")
SRC_TOPDOWN=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['topdown_connections'])")
SRC_SKIP=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['skip_connections'])")
SRC_BIO_UNROLL=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['bio_unroll'])")
SRC_BOTTLENECKS=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d.get('bottlenecks',''))")
SRC_DATASET=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['dataset'])")
SRC_DATASET_MODE=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['dataset_mode'])")
SRC_BATCH_SIZE=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['batch_size'])")
SRC_BATCH_SIZE_VT=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['batch_size_val_test'])")
SRC_N_EPOCHS=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['n_epochs'])")
SRC_LR=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['learning_rate'])")
SRC_NUM_WORKERS=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d['num_workers'])")
SRC_GRAD_CLIP=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d.get('grad_clipping',1))")
SRC_NAME=$(python -c "import json; d=json.load(open('$ARGS_JSON')); print(d.get('name',''))")

# Apply overrides (use override if non-empty, else keep source)
DATASET="${OVERRIDE_DATASET:-$SRC_DATASET}"
NAME="${OVERRIDE_NAME:-$SRC_NAME}"
BATCH_SIZE="${OVERRIDE_BATCH_SIZE:-$SRC_BATCH_SIZE}"
BATCH_SIZE_VT="${OVERRIDE_BATCH_SIZE_VAL_TEST:-$SRC_BATCH_SIZE_VT}"
N_EPOCHS="${OVERRIDE_N_EPOCHS:-$SRC_N_EPOCHS}"
LR="${OVERRIDE_LEARNING_RATE:-$SRC_LR}"
NUM_WORKERS="${OVERRIDE_NUM_WORKERS:-$SRC_NUM_WORKERS}"

echo "====================================="
echo "Training BLT-VS on EcoSet"
echo "  Source run:  $SOURCE_RUN"
echo "  Network:     $SRC_NETWORK"
echo "  Bottlenecks: $SRC_BOTTLENECKS"
echo "  Timesteps:   $SRC_TIMESTEPS"
echo "  Dataset:     $DATASET (was: $SRC_DATASET)"
echo "  Batch size:  $BATCH_SIZE"
echo "  LR:          $LR"
echo "  Epochs:      $N_EPOCHS"
echo "  Name:        $NAME"
echo "  Start time:  $(date)"
echo "====================================="

python blt_vs_model/training_code/train_net_copy_hooks.py \
    --network "$SRC_NETWORK" \
    --bottlenecks "$SRC_BOTTLENECKS" \
    --name "$NAME" \
    --dataset_mode "$SRC_DATASET_MODE" \
    --dataset "$DATASET" \
    --timesteps "$SRC_TIMESTEPS" \
    --lateral_connections "$SRC_LATERAL" \
    --topdown_connections "$SRC_TOPDOWN" \
    --skip_connections "$SRC_SKIP" \
    --bio_unroll "$SRC_BIO_UNROLL" \
    --batch_size "$BATCH_SIZE" \
    --batch_size_val_test "$BATCH_SIZE_VT" \
    --n_epochs "$N_EPOCHS" \
    --learning_rate "$LR" \
    --num_workers "$NUM_WORKERS" \
    --grad_clipping "$SRC_GRAD_CLIP" \
    --grad_accum_steps 4

echo "====================================="
echo "Finished: $(date)"
echo "====================================="
