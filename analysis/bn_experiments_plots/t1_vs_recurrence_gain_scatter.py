"""
Scatter plot: readout timestep-1 accuracy (x) vs. recurrence gain (y),
evaluated at each model's best validation-accuracy epoch.

Best epoch      = argmax over epochs of (max accuracy across timesteps).
t1 performance  = val_accuracies_all[best_epoch, 0].
Recurrence gain = max_t val_accuracies_all[best_epoch, :] - t1 performance.

Models are grouped into bottleneck families (V1V2, V2V3, BNall), each swept
over several ranks, plus the no-bottleneck baseline. Points within a family are
connected (sorted by rank) to show the trajectory as the bottleneck tightens.

Data source: loss_*.npz files containing 'val_accuracies_all' (epochs x timesteps).
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURE HERE
# ============================================================

BU_ROOT = r"C:\Users\moehl\Logs\Final\BU"

# Each family: display name, color, marker, and (rank -> variant folder) builder.
# Only ranks that actually exist on disk are used.
FAMILIES = [
    {
        "name": "V1V2",
        "color": "#2a9d8f",
        "marker": "o",
        "parent": os.path.join(BU_ROOT, "BNV1V2_BU"),
        "folder": lambda r: f"BNV1V2_BU_{r}",
        "ranks": [12, 32, 64, 128, 192],
    },
    {
        "name": "V2V3",
        "color": "#e76f51",
        "marker": "s",
        "parent": os.path.join(BU_ROOT, "BNV2V3_BU"),
        "folder": lambda r: f"BNV2V3_BU_{r}",
        "ranks": [8, 12, 32, 64, 128, 256],
    },
    {
        "name": "BNall",
        "color": "#9b5de5",
        "marker": "^",
        "parent": os.path.join(BU_ROOT, "BNall_BU"),
        "folder": lambda r: f"bnall{r}_BU",
        "ranks": [64, 96],
    },
]

# No-bottleneck baseline
BASELINE = (
    r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800.npz",
    "BNnone",
)

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

family_points = {}  # name -> list of (rank, t1, gain)
for fam in FAMILIES:
    pts = []
    for r in fam["ranks"]:
        variant_dir = os.path.join(fam["parent"], fam["folder"](r))
        if not os.path.isdir(variant_dir):
            print(f"WARNING: missing folder {variant_dir}, skipping")
            continue
        npz_path = find_loss_npz(variant_dir)
        if npz_path is None:
            print(f"WARNING: no loss_*.npz in {variant_dir}, skipping")
            continue
        res = t1_and_gain_at_best_epoch(npz_path)
        if res is None:
            print(f"WARNING: no val_accuracies_all in {npz_path}, skipping")
            continue
        t1_acc, gain = res
        pts.append((r, t1_acc, gain))
        print(f"{fam['name']}-{r}: t1={t1_acc:.2f}%, gain={gain:.2f}pp")
    pts.sort(key=lambda x: x[0])  # sort by rank
    family_points[fam["name"]] = pts

# Baseline
base_pt = None
if os.path.exists(BASELINE[0]):
    res = t1_and_gain_at_best_epoch(BASELINE[0])
    if res is not None:
        base_pt = res
        print(f"{BASELINE[1]}: t1={res[0]:.2f}%, gain={res[1]:.2f}pp")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6.5))

for fam in FAMILIES:
    pts = family_points.get(fam["name"], [])
    if not pts:
        continue
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]

    # Faint trajectory line (rank decreasing = tighter bottleneck)
    ax.plot(xs, ys, color=fam["color"], alpha=0.35, linewidth=1.5, zorder=2)
    ax.scatter(xs, ys, s=130, color=fam["color"], marker=fam["marker"],
               edgecolor="black", linewidth=0.8, zorder=3, label=fam["name"])

    for r, x, y in pts:
        ax.annotate(str(r), (x, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=fam["color"], fontweight="bold")

# Baseline point
if base_pt is not None:
    ax.scatter(base_pt[0], base_pt[1], s=220, color="black", marker="*",
               edgecolor="white", linewidth=0.8, zorder=4, label="BNnone (baseline)")
    ax.annotate("none", (base_pt[0], base_pt[1]), textcoords="offset points",
                xytext=(8, -12), fontsize=9, color="black", fontweight="bold")

ax.set_xlabel("Readout timestep-1 accuracy (%)", fontsize=12)
ax.set_ylabel("Recurrence gain (best timestep - t1, pp)", fontsize=12)
ax.set_title("First-timestep performance vs. recurrence gain\n"
             "(at best validation-accuracy epoch; numbers = bottleneck rank)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right", title="Bottleneck family")

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
