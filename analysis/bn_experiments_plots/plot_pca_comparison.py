"""
PCA 95% Dimensionality Comparison — one subplot per visual area.

Shows how many channels are needed to explain 95% of variance
at each timestep, for each model. Reveals how bottlenecks change
representation compression across the processing hierarchy.
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_npz(log_dir):
    """Return the pca_results_streaming.npz in log_dir, or None."""
    if os.path.isfile(log_dir):
        return log_dir
    pca_path = os.path.join(log_dir, "pca_results_streaming.npz")
    if os.path.exists(pca_path):
        return pca_path
    # fallback: any npz with pca in name
    matches = glob.glob(os.path.join(log_dir, "*pca*.npz"))
    if not matches:
        print(f"WARNING: No pca_results_streaming.npz found in {log_dir}, skipping.")
        return None
    return matches[0]

# ============================================================
# CONFIGURE HERE
# ============================================================

MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNnone_BU_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523", "BNnone_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNall_BU_Skip\BNall64_BU_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-bnall64skip__20260416_130242", "BNall64_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNnone_BU_TD\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158", "BNnone_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNall_BU_TD\BNall64_BU_TD\blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-TD__20260422_112005", "BNall64_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNnone_BU_TD_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019", "BNnone_BU_ TD_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNall_BU_TD_Skip\BNall32_BU_TD_Skip\blt_vs_bottleneck__miniecoset__ts12__bnall32_BU-TD-Skip__20260602_005408", "BNall32_BU_ TD_Skip"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\pca_95_comparison_ablation.png"
SAVE_PATH_DIFF = r"C:\Users\moehl\Logs\Plots_BA\pca_95_difference_ablation.png"
LEVEL = 95
N_TIMESTEPS = 12

# ============================================================
# CONSTANTS
# ============================================================

AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

TOTAL_CHANNELS = {
    "Retina": 32,
    "LGN": 32,
    "V1": 576,
    "V2": 480,
    "V3": 352,
    "V4": 256,
    "LOC": 352,
}

COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#9b5de5", "#f15bb5"]

# ============================================================
# LOAD DATA
# ============================================================

model_data = []

for log_dir, label in MODELS:
    npz_path = find_npz(log_dir)
    if npz_path is None:
        continue

    data = np.load(npz_path)
    area_curves = {}

    for area in AREAS:
        dims = []
        for t in range(N_TIMESTEPS):
            key = f"{area}_t{t}_channels_{LEVEL}"
            dims.append(int(data[key][0]) if key in data else 0)
        area_curves[area] = np.array(dims)

    model_data.append({"label": label, "curves": area_curves})
    print(f"Loaded: {label}")

if not model_data:
    print("ERROR: No models loaded.")
    exit(1)

# ============================================================
# PLOT — one subplot per area
# ============================================================

n_areas = len(AREAS)
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes_flat = axes.flatten()

timesteps = np.arange(N_TIMESTEPS)

for idx, area in enumerate(AREAS):
    ax = axes_flat[idx]
    total = TOTAL_CHANNELS[area]

    for i, md in enumerate(model_data):
        curve = md["curves"][area]
        ax.plot(timesteps, curve, marker="o", markersize=4, linewidth=2,
                color=COLORS[i % len(COLORS)], label=md["label"])

    ax.axhline(total, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(N_TIMESTEPS - 0.5, total, f"max={total}", va="bottom", ha="right",
            fontsize=7, color="gray")

    ax.set_title(area, fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel(f"Channels ({LEVEL}%)", fontsize=8)
    ax.set_xticks(timesteps)
    ax.set_xlim(-0.3, N_TIMESTEPS - 0.7)
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)

# Remove unused subplot (2x4 = 8 slots, 7 areas)
axes_flat[-1].set_visible(False)

# Shared legend in the empty slot
handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[-1].set_visible(True)
axes_flat[-1].axis("off")
for spine in axes_flat[-1].spines.values():
    spine.set_visible(False)
axes_flat[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)

fig.suptitle(f"Effective Dimensionality ({LEVEL}% variance) — Model Comparison",
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved: {SAVE_PATH}")
plt.close()

# ============================================================
# PLOT 2 — Difference relative to first model (BN-none)
# ============================================================

baseline = model_data[0]
others = model_data[1:]

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes2_flat = axes2.flatten()

for idx, area in enumerate(AREAS):
    ax = axes2_flat[idx]
    base_curve = baseline["curves"][area]

    for i, md in enumerate(others):
        diff = md["curves"][area] - base_curve
        ax.plot(timesteps, diff, marker="o", markersize=4, linewidth=2,
                color=COLORS[(i + 1) % len(COLORS)], label=md["label"])

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title(area, fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel(f"Δ Channels vs {baseline['label']}", fontsize=8)
    ax.set_xticks(timesteps)
    ax.set_xlim(-0.3, N_TIMESTEPS - 0.7)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)

# Legend in empty slot
axes2_flat[-1].axis("off")
for spine in axes2_flat[-1].spines.values():
    spine.set_visible(False)
handles2, labels2 = axes2_flat[0].get_legend_handles_labels()
axes2_flat[-1].legend(handles2, labels2, loc="center", fontsize=10, frameon=False)

fig2.suptitle(f"Δ Effective Dimensionality ({LEVEL}%) vs {baseline['label']}",
              fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(SAVE_PATH_DIFF, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH_DIFF}")
plt.close()
