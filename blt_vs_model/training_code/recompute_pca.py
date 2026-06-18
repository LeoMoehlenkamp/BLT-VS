"""
Standalone PCA Recomputation Script

Recomputes PCA statistics and plots for an already-trained model.
Uses the corrected compute_first_signal logic that accounts for skip connections.

Usage (from blt_vs_model/training_code/):
    python recompute_pca.py --model_name "PCA1_allBN64_skip__20260417_143012"

    # Optional: use LAST instead of BEST weights
    python recompute_pca.py --model_name "..." --use_best 0

    # Optional: more batches for stable PCA
    python recompute_pca.py --model_name "..." --max_batches 100
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.helper_funcs import get_Dataset_loaders, compute_first_signal
from models.helper_funcs import get_network_model

# ============================
# ARGUMENTS
# ============================

parser = argparse.ArgumentParser(description="Recompute PCA for a trained model")
parser.add_argument("--model_name", type=str, required=True,
                    help="Name of the model folder (in logs/perf_logs/ and logs/net_params/)")
parser.add_argument("--use_best", type=int, default=1,
                    help="1 = load BEST weights, 0 = load LAST weights")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to run on (cuda or cpu)")

args = parser.parse_args()

MODEL_NAME = args.model_name
DEVICE = args.device

# ============================
# PATHS
# ============================

log_path = os.path.join("logs", "perf_logs", MODEL_NAME)
net_path = os.path.join("logs", "net_params", MODEL_NAME)

config_path = os.path.join(log_path, "config.json")

if not os.path.exists(config_path):
    print(f"ERROR: config.json not found at {config_path}")
    sys.exit(1)

# ============================
# LOAD CONFIG
# ============================

with open(config_path, "r") as f:
    hyp = json.load(f)

print(f"Loaded config from {config_path}")
print(f"  Network: {hyp['network']['name']}")
print(f"  Dataset: {hyp['dataset']['name']}")
print(f"  Timesteps: {hyp['network']['timesteps']}")
print(f"  Bottlenecks: {hyp['network'].get('bottlenecks', {})}")
print(f"  Skip connections: {hyp['network'].get('skip_connections', 0)}")

# Use device from argument, all other settings come from config (same as training)
hyp["optimizer"]["device"] = DEVICE

# ============================
# LOAD MODEL
# ============================

model, _ = get_network_model(hyp)

# Find weight file
weight_files = os.listdir(net_path)
if args.use_best:
    candidates = [f for f in weight_files if "BEST" in f]
else:
    candidates = [f for f in weight_files if "LAST" in f]

if not candidates:
    print(f"ERROR: No {'BEST' if args.use_best else 'LAST'} weights found in {net_path}")
    print(f"  Available: {weight_files}")
    sys.exit(1)

weight_path = os.path.join(net_path, candidates[0])
print(f"Loading weights: {weight_path}")

state_dict = torch.load(weight_path, map_location="cpu")
# Handle DataParallel saved weights
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# Filter FLOP keys
state_dict = {k: v for k, v in state_dict.items()
              if "total_ops" not in k and "total_params" not in k}

model.load_state_dict(state_dict)
model = model.float().to(DEVICE)
model.eval()
print("Model loaded and ready.")

# ============================
# LOAD VALIDATION DATA
# ============================

_, val_loader, _, hyp = get_Dataset_loaders(hyp, ["val"])
print(f"Validation batches: {len(val_loader)}")

# ============================
# COMPUTE FIRST SIGNAL
# ============================

bottlenecks = hyp["network"].get("bottlenecks", {})
skip_connections = hyp["network"].get("skip_connections", 0)

first_signal = compute_first_signal(bottlenecks, skip_connections)
print(f"Computed first_signal: {first_signal}")

# ============================
# STREAMING PCA
# ============================

areas_to_extract = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]
timesteps_to_extract = list(range(hyp["network"]["timesteps"]))

cov_mats = {}
sum_vecs = {}
counts = {}

threshold = 1e-8
extract_batches = 0

max_extract_batches = 50
SUB_BATCH_SIZE = 32  # Process in smaller chunks to avoid OOM
print(f"\nExtracting PCA statistics ({max_extract_batches} batches, sub-batch={SUB_BATCH_SIZE})...")

with torch.no_grad():
    for images, labels in val_loader:

        # Split into sub-batches to reduce peak GPU memory
        for sub_start in range(0, images.shape[0], SUB_BATCH_SIZE):
            sub_imgs = images[sub_start:sub_start + SUB_BATCH_SIZE].to(DEVICE)

            outputs, activations = model(
                sub_imgs,
                extract_actvs=True,
                areas=areas_to_extract,
                timesteps=timesteps_to_extract,
            )

            if extract_batches == 0 and sub_start == 0:
                print(f"  Activation dict areas: {list(activations.keys())}")
                for a in activations:
                    print(f"  {a}: timesteps={sorted(activations[a].keys())}, type={type(next(iter(activations[a].values()), None))}")

            for area in activations:
                for t in activations[area]:

                    act = activations[area][t]

                    if act is None:
                        if extract_batches == 0 and sub_start == 0:
                            print(f"  {area} t{t}: act is None (skipped)")
                        continue

                    if isinstance(act, dict):
                        act = next(iter(act.values()))

                    # Skip timesteps before signal arrival
                    if area in first_signal and t < first_signal[area]:
                        max_val = act.abs().max().item()
                        if extract_batches == 0 and sub_start == 0:
                            mean_val = act.abs().mean().item()
                            print(f"  {area} t{t}: max={max_val:.2e}, mean={mean_val:.2e} (skipped, pre-signal)")
                        if max_val > threshold:
                            print(f"  ⚠ Unexpected large activation at {area} t{t}")
                        continue

                    key = f"{area}_t{t}"

                    if extract_batches == 0 and sub_start == 0:
                        max_val = act.abs().max().item()
                        mean_val = act.abs().mean().item()
                        print(f"  {area} t{t}: shape={list(act.shape)}, max={max_val:.2e}, mean={mean_val:.2e} (PROCESSED)")

                    # Spatial subsampling
                    act = act[:, :, ::2, ::2]
                    B, C, H, W = act.shape
                    X = act.permute(0, 2, 3, 1).reshape(-1, C).detach().float()

                    if key not in cov_mats:
                        cov_mats[key] = torch.zeros(C, C, device=X.device, dtype=torch.float32)
                        sum_vecs[key] = torch.zeros(C, device=X.device, dtype=torch.float32)
                        counts[key] = 0

                    cov_mats[key] += X.T @ X
                    sum_vecs[key] += X.sum(dim=0)
                    counts[key] += X.shape[0]

            # Free GPU memory after each sub-batch
            del outputs, activations
            torch.cuda.empty_cache()

        extract_batches += 1
        if extract_batches >= max_extract_batches:
            break

print(f"\nFinished accumulating covariance matrices ({extract_batches} batches).")
print(f"Keys in cov_mats ({len(cov_mats)}): {sorted(cov_mats.keys())}")

# ============================
# COMPUTE PCA RESULTS
# ============================

pca_results = {}

for key in sorted(cov_mats.keys()):

    n = counts[key]
    mean = sum_vecs[key] / n
    cov = (cov_mats[key] / n) - torch.outer(mean, mean)
    cov = cov.cpu().numpy()

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    total_var = eigvals.sum()
    if total_var <= 0:
        explained = np.zeros_like(eigvals)
    else:
        explained = eigvals / total_var

    cumulative = np.cumsum(explained)

    channels_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    channels_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    channels_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    pca_results[f"{key}_eigvals"] = eigvals
    pca_results[f"{key}_explained"] = explained
    pca_results[f"{key}_cumulative"] = cumulative
    pca_results[f"{key}_channels_90"] = np.array([channels_90])
    pca_results[f"{key}_channels_95"] = np.array([channels_95])
    pca_results[f"{key}_channels_99"] = np.array([channels_99])

    print(f"  {key}: 90%={channels_90}, 95%={channels_95}, 99%={channels_99}, total={len(eigvals)}")

pca_path = os.path.join(log_path, "pca_results_streaming.npz")
np.savez(pca_path, **pca_results)
print(f"\nSaved PCA results to: {pca_path}")

# ============================
# PCA DIMENSIONALITY PLOTS
# ============================

print("Generating PCA plots...")

data = np.load(pca_path)

areas = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]
n_timesteps = hyp["network"]["timesteps"]

total_channels = {
    "Retina": 32,
    "LGN": 32,
    "V1": 576,
    "V2": 480,
    "V3": 352,
    "V4": 256,
    "LOC": 352,
}

levels = [90, 95, 99]

for level in levels:

    dim_matrix = []

    for area in areas:
        row = []
        for t in range(n_timesteps):
            key = f"{area}_t{t}_channels_{level}"
            if key in data:
                row.append(data[key][0])
            else:
                row.append(0)
        row.append(total_channels[area])
        dim_matrix.append(row)

    dim_matrix = np.array(dim_matrix)
    heatmap_abs = dim_matrix[:, :-1]
    totals = np.array([total_channels[a] for a in areas])[:, None]
    heatmap_rel = heatmap_abs / totals

    fig, axes = plt.subplots(
        2, 2,
        figsize=(22, 10),
        gridspec_kw={"height_ratios": [1, 0.65], "wspace": 0.35, "hspace": 0.12},
    )

    # Absolute heatmap
    ax = axes[0, 0]
    im = ax.imshow(heatmap_abs, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Channels for {level}% variance")
    ax.set_xticks(range(n_timesteps))
    ax.set_xticklabels(range(n_timesteps))
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Visual Area")
    ax.set_title(f"Representation Dimensionality ({level}% variance)")

    # Relative heatmap
    ax = axes[0, 1]
    im = ax.imshow(heatmap_rel, aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction of total channels")
    ax.set_xticks(range(n_timesteps))
    ax.set_xticklabels(range(n_timesteps))
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Visual Area")
    ax.set_title(f"Relative Dimensionality ({level}% variance)")

    # Absolute table
    ax = axes[1, 0]
    ax.axis("off")
    table_abs = ax.table(
        cellText=dim_matrix,
        rowLabels=areas,
        colLabels=[f"t{i}" for i in range(n_timesteps)] + ["Total"],
        cellLoc="center",
        bbox=[0, 0.20, 1, 0.75],
    )
    table_abs.auto_set_font_size(False)
    table_abs.set_fontsize(11)
    table_abs.scale(1.2, 1.6)

    # Relative table
    ax = axes[1, 1]
    ax.axis("off")
    rel_matrix = np.round(heatmap_rel * 100, 1)
    rel_matrix = np.concatenate([rel_matrix, np.full((len(areas), 1), 100)], axis=1)
    table_rel = ax.table(
        cellText=rel_matrix,
        rowLabels=areas,
        colLabels=[f"t{i}" for i in range(n_timesteps)] + ["Total"],
        cellLoc="center",
        bbox=[0, 0.20, 1, 0.75],
    )
    table_rel.auto_set_font_size(False)
    table_rel.set_fontsize(11)
    table_rel.scale(1.2, 1.6)

    plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05)

    save_file = os.path.join(log_path, f"pca_dimensionality_{level}.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"\nPCA plots saved to: {log_path}")
print("Done.")
