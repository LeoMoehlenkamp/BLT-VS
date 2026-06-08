"""
Plot t=1 validation accuracy over epochs for multiple models.
Each model gets its own subplot with:
  - Solid curve: t=1 accuracy per epoch
  - Horizontal dashed line: best validation accuracy (labeled)
  - Annotated gap (Δ) between best val acc and final t=1 acc

All subplots share the same x-axis (truncated to shortest model)
and y-axis range for fair comparison.

Configure MODELS below: list of (npz_path, label) tuples.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURE HERE — add/remove models as needed
# ============================================================

MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800.npz", "BN-none"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV1V2_BU\BNV1V2_BU_12\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846\loss_blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846.npz", "BNV1V2_BU_12"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV2V3_BU\BNV2V3_BU_12\blt_vs_bottleneck__miniecoset__ts12__bn-V2V3-12__20260328_204839\loss_blt_vs_bottleneck__miniecoset__ts12__bn-V2V3-12__20260328_204839.npz", "BNV2V3_BU_12"),
    (r"C:\Users\moehl\Logs\Final\BU\BNall_BU\BNALL6~1\BLT_VS~1\LOSS_B~1.NPZ", "BNall64"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\t1_vs_best_accuracy.png"

# ============================================================
# LOAD DATA
# ============================================================

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf"]

model_data = []

for npz_path, label in MODELS:
    if not os.path.exists(npz_path):
        print(f"WARNING: File not found, skipping: {npz_path}")
        continue

    data = np.load(npz_path)

    if "val_accuracies_all" not in data:
        print(f"WARNING: 'val_accuracies_all' not found in {npz_path}, skipping")
        continue

    val_all = data["val_accuracies_all"]  # shape: (epochs, timesteps)
    model_data.append((val_all, label))

if len(model_data) == 0:
    print("ERROR: No valid models loaded.")
    exit(1)

# Truncate all to the shortest number of epochs
min_epochs = min(v.shape[0] for v, _ in model_data)

# Global y-axis limits across all models
global_ymin = float("inf")
global_ymax = float("-inf")

for val_all, _ in model_data:
    t1_acc = val_all[:min_epochs, 0]
    best_val = val_all[:min_epochs].max()
    global_ymin = min(global_ymin, t1_acc.min())
    global_ymax = max(global_ymax, best_val)

y_pad = (global_ymax - global_ymin) * 0.1
global_ymin -= y_pad
global_ymax += y_pad

# ============================================================
# PLOT — 2×3 grid layout
# ============================================================

NCOLS = 2
NROWS = 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(5 * NCOLS, 5 * NROWS), sharey=True, sharex=True)
axes_flat = axes.flatten()

for i, (val_all, label) in enumerate(model_data):
    ax = axes_flat[i]
    val_all = val_all[:min_epochs]
    epochs = np.arange(1, min_epochs + 1)

    t1_acc = val_all[:, 0]
    best_val_acc = val_all.max()
    final_t1 = t1_acc[-1]
    gap = best_val_acc - final_t1

    color = COLORS[i % len(COLORS)]

    # Solid curve: t=1 accuracy
    ax.plot(epochs, t1_acc, color=color, linewidth=1.8, label="t=1 accuracy")

    # Dashed line: best val acc
    ax.axhline(y=best_val_acc, color=color, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(
        min_epochs * 0.02, best_val_acc + (global_ymax - global_ymin) * 0.02,
        f"Best val: {best_val_acc:.1f}%",
        color=color, fontsize=8, va="bottom", ha="left", fontweight="bold"
    )

    # Annotate the gap with a double-arrow at the last epoch
    arrow_x = min_epochs - 1
    ax.annotate(
        "", xy=(arrow_x, best_val_acc), xytext=(arrow_x, final_t1),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2)
    )
    ax.text(
        arrow_x - min_epochs * 0.05,
        (best_val_acc + final_t1) / 2,
        f"Δ = {gap:.1f}%",
        fontsize=9, va="center", ha="right", fontweight="bold"
    )

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylim(global_ymin, global_ymax)
    ax.grid(True, alpha=0.3)

    # X label only on bottom row
    if i >= NCOLS:
        ax.set_xlabel("Epoch", fontsize=10)

    # Y label only on left column
    if i % NCOLS == 0:
        ax.set_ylabel("Validation Accuracy (%)", fontsize=10)

# Handle empty subplot(s) — turn off axes, optionally add a legend/note
for j in range(len(model_data), NROWS * NCOLS):
    ax = axes_flat[j]
    ax.set_visible(False)

fig.suptitle("First-Timestep Accuracy vs. Best Overall Accuracy", fontsize=13, y=1.02)
plt.tight_layout()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
