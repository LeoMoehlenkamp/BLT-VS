"""
General box-and-arrow diagram showing how the BLT-VS bottleneck works.

Uses generic labels (Area A / Area B, in_ch / rank / out_ch) instead of a
concrete V1→V2 example.  Box heights are proportional to channel count so the
squeeze-and-expand shape is immediately visible.

Flow (left → right):
  Area A [in_ch]  →  Conv2d(in_ch, rank, k=1)  →  ReLU  →  Conv2d(rank, out_ch, k=k)  →  Area B [out_ch]
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── output path ──────────────────────────────────────────────────────────────
OUT_DIR = r"C:\Users\moehl\Logs\Plots_BA"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "bottleneck_diagram_general.png")

# ── generic channel sizes ────────────────────────────────────────────────────
IN_CH  = 100   # relative unit for source area (= "large")
RANK   = 20    # bottleneck rank    (= "small")
OUT_CH = 80    # receiving area     (= "large but different")

# ── layout ───────────────────────────────────────────────────────────────────
STAGE_X = [0.0, 2.8, 5.0, 7.2, 10.0]   # centre x per stage
STAGE_W = [1.8, 1.9, 1.5, 2.1,  1.8]   # box width per stage

MAX_H = 4.0
MIN_H = 0.6

def ch_to_h(ch):
    return MIN_H + (ch / IN_CH) * (MAX_H - MIN_H)

STAGE_H = [
    ch_to_h(IN_CH),
    ch_to_h(RANK),
    ch_to_h(RANK),
    ch_to_h(OUT_CH),
    ch_to_h(OUT_CH),
]

STAGE_LABELS = [
    "Area A",
    "Conv2d\n(in_ch → rank,\nk=1, s=1)",
    "ReLU",
    "Conv2d\n(rank → out_ch,\nk=k, s=s)",
    "Area B",
]

STAGE_SUBLABELS = [
    "in_ch channels",
    "→ rank channels\n(compressed)",
    "rank channels",
    "→ out_ch channels\n(rescaled)",
    "out_ch channels",
]

# ── colours ──────────────────────────────────────────────────────────────────
C_AREA    = "#4A7FC1"
C_BN_CONV = "#D4822A"
C_RELU    = "#C0392B"
C_BU_CONV = "#27AE60"

STAGE_COLORS = [C_AREA, C_BN_CONV, C_RELU, C_BU_CONV, C_AREA]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig_w = STAGE_X[-1] + STAGE_W[-1] / 2 + 1.5
fig, ax = plt.subplots(figsize=(fig_w, 6.5))

# White background
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_min = STAGE_X[0]  - STAGE_W[0]  / 2
x_max = STAGE_X[-1] + STAGE_W[-1] / 2
x_pad = 1.0
ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(-2.8, MAX_H / 2 + 3.0)
ax.set_aspect("equal", adjustable="datalim")
ax.axis("off")

# Centre the content by computing the mid-x of all boxes and offsetting
x_centre = (x_min + x_max) / 2
ax.set_xlim(x_centre - (x_max - x_min) / 2 - x_pad,
            x_centre + (x_max - x_min) / 2 + x_pad)

ax.set_title(
    "How a BLT-VS Bottleneck Connection Works",
    color="black", fontsize=13, fontweight="bold", pad=14,
)

# ── draw boxes ────────────────────────────────────────────────────────────────
for i, (x, w, h, color, label) in enumerate(
        zip(STAGE_X, STAGE_W, STAGE_H, STAGE_COLORS, STAGE_LABELS)):
    y_bot = -h / 2
    y_top =  h / 2

    rect = FancyBboxPatch(
        (x - w / 2, y_bot), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="white", linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(rect)

    # Label inside box if tall enough, else above
    if h >= 0.9:
        ax.text(x, 0, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white",
                zorder=4, multialignment="center")
    else:
        ax.text(x, y_top + 0.18, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#333333",
                zorder=4, multialignment="center")

    # Sub-label below box
    ax.text(x, -h / 2 - 0.28, STAGE_SUBLABELS[i],
            ha="center", va="top", fontsize=8, color="#555555",
            zorder=4, multialignment="center")

# ── arrows between boxes ──────────────────────────────────────────────────────
for i in range(len(STAGE_X) - 1):
    x0 = STAGE_X[i]     + STAGE_W[i]     / 2
    x1 = STAGE_X[i + 1] - STAGE_W[i + 1] / 2
    ax.annotate(
        "", xy=(x1, 0), xytext=(x0, 0),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.20",
            color="#333333", lw=2.0,
        ),
        zorder=5,
    )

# ── channel labels on arrows ─────────────────────────────────────────────────
arrow_channels = ["in_ch", "rank", "rank", "out_ch"]
for i, label in enumerate(arrow_channels):
    x_mid = (STAGE_X[i] + STAGE_W[i] / 2 + STAGE_X[i + 1] - STAGE_W[i + 1] / 2) / 2
    ax.text(x_mid, 0.22, label,
            ha="center", va="bottom", fontsize=8, color="#333333",
            fontstyle="italic", zorder=5)

# ── bracket: Bottleneck module ───────────────────────────────────────────────
bk_x0 = STAGE_X[1] - STAGE_W[1] / 2 - 0.08
bk_x1 = STAGE_X[2] + STAGE_W[2] / 2 + 0.08
bk_y  = MAX_H / 2 + 0.55
tick  = 0.18
ax.plot([bk_x0, bk_x0, bk_x1, bk_x1],
        [bk_y - tick, bk_y, bk_y, bk_y - tick],
        color=C_BN_CONV, lw=1.8, zorder=3)
ax.text((bk_x0 + bk_x1) / 2, bk_y + 0.12,
        "Bottleneck module\n(compresses information to rank)",
        ha="center", va="bottom", fontsize=8.5, color=C_BN_CONV,
        fontweight="bold", multialignment="center")

# ── bracket: Receiving area's BU conv ────────────────────────────────────────
bu_x0 = STAGE_X[3] - STAGE_W[3] / 2 - 0.08
bu_x1 = STAGE_X[3] + STAGE_W[3] / 2 + 0.08
ax.plot([bu_x0, bu_x0, bu_x1, bu_x1],
        [bk_y - tick, bk_y, bk_y, bk_y - tick],
        color=C_BU_CONV, lw=1.8, zorder=3)
ax.text((bu_x0 + bu_x1) / 2, bk_y + 0.12,
        "Receiving area's\nBU convolution",
        ha="center", va="bottom", fontsize=8.5, color=C_BU_CONV,
        fontweight="bold", multialignment="center")

# ── legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_AREA,    edgecolor="white", label="Cortical area"),
    mpatches.Patch(facecolor=C_BN_CONV, edgecolor="white", label="Conv2d 1×1  (bottleneck compress)"),
    mpatches.Patch(facecolor=C_RELU,    edgecolor="white", label="ReLU activation"),
    mpatches.Patch(facecolor=C_BU_CONV, edgecolor="white", label="Conv2d k×k  (BU conv, rescales to out_ch)"),
]
ax.legend(
    handles=legend_elements,
    loc="lower center", bbox_to_anchor=(0.5, -0.42),
    ncol=2,
    facecolor="white", edgecolor="#AAAAAA",
    labelcolor="#333333", fontsize=8.5,
    framealpha=1.0,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PATH}")
