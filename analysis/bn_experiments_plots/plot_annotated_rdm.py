"""
Plot one first-order RDM (LOC, last timestep) from BNnone_BU with semantic
category annotations, so you can see where categories (human/body-parts,
mammals, food, tools, ...) cluster along the diagonal.

The RDM rows/cols are already ordered by the THINGS taxonomy (sort_idx), and
rdm_row_labels.csv (exported from the cluster) gives the category info per row
in that exact order.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIGURE
# ============================================================

RDM_NPZ = r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms.npz"
RDM_KEY = "LOC_t11_rdm_cosine_ranked"          # last layer (LOC), last timestep
LABELS_CSV = os.path.join(os.path.dirname(__file__), "rdm_row_labels.csv")
SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\rdm_LOC_t11_annotated.png"

CMAP = "viridis"

# Superordinate groups, in assignment priority order.
# (display name, color, predicate over a row dict of the labels csv)
GROUPS = [
    ("Human / body parts", "#e63946", lambda r: r["human"] == 1 or r["body_parts"] == 1),
    ("Mammals",            "#f4a261", lambda r: r["mammal"] == 1),
    ("Non-mammal animals", "#e9c46a", lambda r: r["non_mammal"] == 1),
    ("Food",               "#2a9d8f", lambda r: r["food"] == 1 or r["fruit"] == 1 or r["vegetable"] == 1 or r["other_food"] == 1),
    ("Plants",             "#43aa8b", lambda r: r["plants"] == 1),
    ("Other natural",      "#4d908e", lambda r: r["other_natural"] == 1 or r["natural"] == 1),
    ("Tools / small artificial", "#577590", lambda r: r["tools"] == 1 or r["artificial_small"] == 1 or r["artificial_small_other"] == 1),
    ("Furniture",          "#9b5de5", lambda r: r["furniture"] == 1),
    ("Vehicles",           "#f15bb5", lambda r: r["vehicles"] == 1),
    ("Large / outdoor artificial", "#3a0ca3", lambda r: r["artificial_large"] == 1 or r["outside_large"] == 1),
    ("Other",              "#adb5bd", lambda r: True),
]

# ============================================================
# LOAD
# ============================================================

data = np.load(RDM_NPZ, allow_pickle=True)
rdm = data[RDM_KEY].astype(np.float64)
n = rdm.shape[0]

with open(LABELS_CSV, newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == n, f"labels ({len(rows)}) != rdm rows ({n})"

# convert taxonomy columns to int
TAX_COLS = ["animate", "body_parts", "human", "mammal", "non_mammal", "inanimate",
            "natural", "food", "fruit", "vegetable", "other_food", "plants",
            "other_natural", "artificial", "artificial_small", "tools",
            "artificial_small_other", "artificial_large", "furniture",
            "vehicles", "outside_large"]
for r in rows:
    for c in TAX_COLS:
        if c in r:
            r[c] = int(float(r[c]))


def assign_group(row):
    for name, _color, pred in GROUPS:
        if pred(row):
            return name
    return "Other"


group_of_row = np.array([assign_group(r) for r in rows])

color_map = {name: color for name, color, _ in GROUPS}
row_colors = np.array([color_map[g] for g in group_of_row])

# Contiguous segments of the same group -> for boundaries + centered labels
segments = []  # (group, start, end)
start = 0
for i in range(1, n + 1):
    if i == n or group_of_row[i] != group_of_row[start]:
        segments.append((group_of_row[start], start, i))
        start = i

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 11))

im = ax.imshow(rdm, cmap=CMAP, aspect="equal", extent=[0, n, n, 0])

# Category boundary lines (exactly on the RDM, in data coords 0..n),
# including the outer edges, so each category is clearly delimited.
edges = [0] + [s for _g, s, _e in segments if s != 0] + [n]
for b in edges:
    ax.axhline(b, color="black", linewidth=1.3)
    ax.axvline(b, color="black", linewidth=1.3)

# Category names as ticks at block centres (fits width/height exactly)
centres = [(s + e) / 2 for _g, s, e in segments]
names = [g for g, _s, _e in segments]
ax.set_xticks(centres)
ax.set_xticklabels(names, rotation=90, fontsize=9)
ax.set_yticks(centres)
ax.set_yticklabels(names, fontsize=9)
ax.tick_params(length=0)

# Keep the axes tight to the RDM extent
ax.set_xlim(0, n)
ax.set_ylim(n, 0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Dissimilarity (ranked cosine distance)", fontsize=10)

ax.set_title("BNnone_BU  -  LOC, t11  (first-order RDM, cosine ranked)",
             fontsize=13, fontweight="bold", pad=10)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")

# quick summary of block sizes
print("\nCategory blocks (in RDM order):")
for g, s, e in segments:
    print(f"  {g:32s} rows {s:4d}-{e-1:4d}  (n={e-s})")
plt.close()
