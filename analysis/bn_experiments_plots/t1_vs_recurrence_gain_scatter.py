"""
Scatter plot: readout timestep-1 accuracy (x) vs. recurrence gain (y),
evaluated at each model's best validation-accuracy epoch.

Best epoch      = argmax over epochs of (max accuracy across timesteps).
t1 performance  = val_accuracies_all[best_epoch, 0].
Recurrence gain = max_t val_accuracies_all[best_epoch, :] - t1 performance.

Dashed diagonal lines show iso-final-performance contours (x + y = constant).

Data source: loss_*.npz files containing 'val_accuracies_all' (epochs x timesteps).
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURE HERE
# ============================================================

BU_ROOT = r"C:\Users\moehl\Logs\Final\BU"

# Individual models to show.  Each entry: display label, color, marker, variant dir.
MODELS = [
    {
        "label": "BNnone_BU",
        "color": "#1a5276",
        "marker": "o",
        "dir": None,  # uses NPZ_PATH directly
        "npz": r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800.npz",
    },
    {
        "label": "BNV1V2_BU_192",
        "color": "#17a589",
        "marker": "o",
        "dir": os.path.join(BU_ROOT, "BNV1V2_BU", "BNV1V2_BU_192"),
        "npz": None,
    },
    {
        "label": "BNV1V2_BU_32",
        "color": "#d4ac0d",
        "marker": "o",
        "dir": os.path.join(BU_ROOT, "BNV1V2_BU", "BNV1V2_BU_32"),
        "npz": None,
    },
    {
        "label": "BNV1V2_BU_12",
        "color": "#e67e22",
        "marker": "o",
        "dir": os.path.join(BU_ROOT, "BNV1V2_BU", "BNV1V2_BU_12"),
        "npz": None,
    },
    {
        "label": "BNV2V3_BU_8",
        "color": "#7d3c98",
        "marker": "o",
        "dir": os.path.join(BU_ROOT, "BNV2V3_BU", "BNV2V3_BU_8"),
        "npz": None,
    },
    {
        "label": "BNall_BU_64",
        "color": "#e91e8c",
        "marker": "o",
        "dir": os.path.join(BU_ROOT, "BNall_BU", "bnall64_BU"),
        "npz": None,
    },
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\t1_vs_recurrence_gain_scatter.png"

# ============================================================
# HELPERS
# ============================================================


def find_loss_npz(variant_dir):
    matches = glob.glob(os.path.join(variant_dir, "**", "loss_*.npz"), recursive=True)
    return matches[0] if matches else None


def t1_and_gain_at_best_epoch(npz_path):
    data = np.load(npz_path)
    if "val_accuracies_all" not in data:
        return None
    val = data["val_accuracies_all"]              # (epochs, timesteps)
    best_epoch = int(np.argmax(val.max(axis=1)))  # best val-acc epoch
    t1_acc = float(val[best_epoch, 0])
    best_val = float(val[best_epoch].max())
    return t1_acc, best_val - t1_acc              # (t1, recurrence gain)

# ============================================================
# LOAD & COMPUTE
# ============================================================

model_points = []  # list of (label, color, marker, t1, gain)
for m in MODELS:
    npz_path = m["npz"] if m["npz"] else find_loss_npz(m["dir"])
    if npz_path is None or not os.path.exists(npz_path):
        print(f"WARNING: could not find npz for {m['label']}, skipping")
        continue
    res = t1_and_gain_at_best_epoch(npz_path)
    if res is None:
        print(f"WARNING: no val_accuracies_all in {npz_path}, skipping")
        continue
    t1_acc, gain = res
    model_points.append((m["label"], m["color"], m["marker"], t1_acc, gain))
    print(f"{m['label']}: t1={t1_acc:.2f}%, gain={gain:.2f}pp")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6.5))

for label, color, marker, x, y in model_points:
    ax.scatter(x, y, s=180, color=color, marker=marker,
               edgecolor="black", linewidth=0.8, zorder=3, label=label)
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(9, 4), fontsize=8, color=color, fontweight="bold")

ax.set_xlabel("Readout timestep-1 accuracy (%)", fontsize=12)
ax.set_ylabel("Recurrence gain (best timestep - t1, pp)", fontsize=12)
ax.set_title("First-timestep performance vs. recurrence gain\n"
             "(at best validation-accuracy epoch)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# --- iso-performance diagonals (x + y = final_acc = constant) ---
xlim = ax.get_xlim()
ylim = ax.get_ylim()

fp_min = xlim[0] + ylim[0]
fp_max = xlim[1] + ylim[1]
fp_step = 2  # percentage points between each diagonal
fp_start = int(np.ceil(fp_min / fp_step)) * fp_step
fp_vals = list(range(fp_start, int(np.floor(fp_max / fp_step)) * fp_step + 1, fp_step))

for fp in fp_vals:
    ax.axline((xlim[0], fp - xlim[0]), slope=-1,
              color="gray", alpha=0.4, linewidth=0.9,
              linestyle="--", zorder=1)
    x_entries = []
    if ylim[0] <= fp - xlim[0] <= ylim[1]:
        x_entries.append(xlim[0])
    if xlim[0] <= fp - ylim[0] <= xlim[1]:
        x_entries.append(fp - ylim[0])
    if ylim[0] <= fp - xlim[1] <= ylim[1]:
        x_entries.append(xlim[1])
    if xlim[0] <= fp - ylim[1] <= xlim[1]:
        x_entries.append(fp - ylim[1])
    if len(x_entries) >= 2:
        x_mid = (x_entries[0] + x_entries[-1]) / 2
        y_mid = fp - x_mid
        ax.text(x_mid, y_mid, f"{fp}%", fontsize=8, color="dimgray",
                ha="center", va="center", clip_on=True,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=1)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Build legend: one entry per model + one entry for the diagonal lines
handles, labels = ax.get_legend_handles_labels()
diag_handle = Line2D([0], [0], color="gray", alpha=0.6, linewidth=1.2,
                     linestyle="--", label="Equal final accuracy (t1 + gain)")
ax.legend(handles=handles + [diag_handle],
          labels=labels + ["Equal final accuracy (t1 + gain)"],
          fontsize=9, loc="upper right")

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()

