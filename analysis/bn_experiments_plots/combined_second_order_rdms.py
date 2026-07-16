"""
Recreate second-order (time-time) RDMs directly from the saved ann_rdms.npz files
and combine them into one big panel with a shared, normalized color scale.

Rows = models, Columns = areas (Retina and LGN excluded).
Each cell is the time-time RDM: 1 - spearman(rdm_t_i, rdm_t_j) across timesteps,
identical to analysis/rdm_generation/second_order_rdms_extended.py.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# ============================================================
# CONFIGURE HERE
# ============================================================

# (path_to_ann_rdms.npz, model_label)
MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms.npz", "BNnone_BU"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV1V2_BU\BNV1V2_BU_12\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846_ann_rdms.npz", "BNV1V2_BU_12"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV2V3_BU\BNV2V3_BU_8\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-V2V3-8__20260329_132907_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-V2V3-8__20260329_132907_ann_rdms.npz", "BNV2V3_BU_8"),
    (r"C:\Users\moehl\Logs\Final\BU\BNall_BU\bnall64_BU\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-RetinaLGN-64_LGNV1-64_V1V2-64_V2V3-64_V3V4-64_V4LOC-64__20260406_113212_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-RetinaLGN-64_LGNV1-64_V1V2-64_V2V3-64_V3V4-64_V4LOC-64__20260406_113212_ann_rdms.npz", "BNall_BU_64"),
]

# Areas to plot (Retina and LGN excluded)
AREAS = ["V1", "V2", "V3", "V4", "LOC"]

# Must match how the source RDMs were saved (cosine_ranked here)
METRIC = "cosine"
RDM_TYPE = "ranked"

# High-contrast colormap; "turbo" / "magma" / "inferno" reveal small differences well.
CMAP = "turbo"
# gamma < 1 stretches the low end so tiny dissimilarities become visible.
# Set to 1.0 for a plain linear scale.
GAMMA = 0.5
SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\second_order_BNstages_combined.png"

# ============================================================
# CORE
# ============================================================


def extract_matching_keys(npz_file, area, metric, rdm_type):
    pattern = re.compile(rf"^{area}_t(\d+)_rdm_{metric}_{rdm_type}$")
    matches = []
    for key in npz_file.files:
        m = pattern.match(key)
        if m:
            matches.append((int(m.group(1)), key))
    matches.sort(key=lambda x: x[0])
    return matches


def compute_time_time_rdm(data, area):
    matches = extract_matching_keys(data, area, METRIC, RDM_TYPE)
    if len(matches) == 0:
        return None, None
    timesteps = [t for t, _ in matches]
    rdms = np.array([squareform(data[key].astype(np.float64), checks=False)
                     for _, key in matches])
    spearman_rs, _ = spearmanr(rdms, axis=1)
    time_time_rdm = 1 - spearman_rs
    return time_time_rdm, timesteps


# Load and compute all RDMs first (needed for a shared color scale)
results = {}  # (model_label) -> {area: (rdm, timesteps)}
for npz_path, label in MODELS:
    if not os.path.exists(npz_path):
        print(f"WARNING: Not found, skipping: {npz_path}")
        continue
    data = np.load(npz_path, allow_pickle=True)
    area_rdms = {}
    for area in AREAS:
        rdm, timesteps = compute_time_time_rdm(data, area)
        if rdm is not None:
            area_rdms[area] = (rdm, timesteps)
    results[label] = area_rdms

if not results:
    print("ERROR: No data loaded.")
    raise SystemExit(1)

# Global normalization across ALL panels so they are comparable
all_vals = np.concatenate([
    rdm.ravel()
    for area_rdms in results.values()
    for rdm, _ in area_rdms.values()
])
vmin = float(np.min(all_vals))
vmax = float(np.max(all_vals))
norm = PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax)
print(f"Shared color scale: vmin={vmin:.4f}, vmax={vmax:.4f}, gamma={GAMMA}")

# ============================================================
# PLOT
# ============================================================

model_labels = list(results.keys())
n_rows = len(model_labels)
n_cols = len(AREAS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.2 * n_cols, 3.2 * n_rows),
                         squeeze=False)

im = None
for r, label in enumerate(model_labels):
    area_rdms = results[label]
    for c, area in enumerate(AREAS):
        ax = axes[r][c]
        if area not in area_rdms:
            ax.axis("off")
            continue
        rdm, timesteps = area_rdms[area]
        im = ax.imshow(rdm, cmap=CMAP, norm=norm)

        ticks = np.arange(len(timesteps))
        labels_t = [f"t{t}" for t in timesteps]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels_t, rotation=90, fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels_t, fontsize=6)

        if r == 0:
            ax.set_title(area, fontsize=12, fontweight="bold")
        if c == 0:
            ax.set_ylabel(label, fontsize=11, fontweight="bold")

# Shared colorbar for the whole figure
fig.subplots_adjust(right=0.90, hspace=0.35, wspace=0.35)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Second-order distance (1 - Spearman)")

plt.suptitle("Second-order RDMs (time x time) - first-order: cosine, rank-transformed",
             fontsize=14, fontweight="bold")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
