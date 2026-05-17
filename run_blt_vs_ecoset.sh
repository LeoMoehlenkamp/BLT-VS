#!/bin/bash
#SBATCH --partition=klab-gpu
#SBATCH -w klab-7
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=0
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
# Try to copy EcoSet H5 to node-local storage for performance.
# If no local disk has enough space, fall back to NFS directly.
# With --mem=0 the cgroup limit = total node RAM, so the kernel
# can properly evict NFS page cache without OOM.
# ============================================================
REMOTE_H5="/share/klab/datasets/ecoset_square256_proper_chunks.h5"
REMOTE_SIZE_KB=$(du -k "$REMOTE_H5" | cut -f1)
NEED_KB=$((REMOTE_SIZE_KB + 10*1024*1024))  # file size + 10GB headroom

# Candidate local storage directories (checked in order)
LOCAL_CANDIDATES=("${TMPDIR:-}" "/local" "/scratch" "/localscratch" "/local_scratch" "/tmp")

CHOSEN_DIR=""
for cand in "${LOCAL_CANDIDATES[@]}"; do
    [ -z "$cand" ] && continue
    [ -d "$cand" ] || continue
    AVAIL_KB=$(df -k "$cand" | tail -1 | awk '{print $4}')
    echo "Probing $cand: $(( AVAIL_KB / 1024 / 1024 ))GB available (need $(( NEED_KB / 1024 / 1024 ))GB)"
    if [ "$AVAIL_KB" -ge "$NEED_KB" ]; then
        CHOSEN_DIR="$cand"
        break
    fi
done

if [ -n "$CHOSEN_DIR" ]; then
    LOCAL_DIR="${CHOSEN_DIR}/ecoset_${SLURM_JOB_ID}"
    LOCAL_H5="${LOCAL_DIR}/ecoset_square256_proper_chunks.h5"

    trap "echo 'Cleaning up local data...'; rm -rf ${LOCAL_DIR}" EXIT

    mkdir -p "$LOCAL_DIR"
    echo "Copying EcoSet to node-local storage..."
    echo "  From: $REMOTE_H5"
    echo "  To:   $LOCAL_H5"
    cp_start=$(date +%s)
    cp "$REMOTE_H5" "$LOCAL_H5"
    cp_rc=$?
    cp_end=$(date +%s)
    echo "Copy finished in $((cp_end - cp_start))s (exit=$cp_rc)"

    if [ $cp_rc -eq 0 ] && [ -f "$LOCAL_H5" ]; then
        DATASET_PATH="${LOCAL_DIR}/"
        echo "Using LOCAL dataset path: $DATASET_PATH"
    else
        echo "WARNING: Copy failed, falling back to NFS."
        rm -rf "$LOCAL_DIR"
        DATASET_PATH="/share/klab/datasets/"
        echo "Using NFS dataset path: $DATASET_PATH"
    fi
else
    echo "No local disk with enough space found. Using NFS directly."
    echo "This works because --mem=0 gives the job all node RAM,"
    echo "so the kernel can evict NFS page cache normally."
    DATASET_PATH="/share/klab/datasets/"
fi
echo "Final dataset path: $DATASET_PATH"

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
    --dataset_path "$DATASET_PATH" \
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
