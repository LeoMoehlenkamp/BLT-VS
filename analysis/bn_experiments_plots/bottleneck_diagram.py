"""
Box-and-arrow diagram showing how a single BLT-VS bottleneck connection works.

Flow (left → right):
  Source area  →  Conv2d(in_ch, rank, k=1)  →  ReLU  →  Conv2d(rank, out_ch, k=k)  →  Receiving area

The bottleneck compresses the source area's activations to a low rank before
the receiving area's BU convolution rescales them to its own channel count.
Box heights are proportional to channel count so the "bottleneck" shape is visible.
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
OUT_PATH = os.path.join(OUT_DIR, "bottleneck_diagram.png")

# ── concrete example: V1 → V2 ────────────────────────────────────────────────
SRC_NAME  = "V1\n(source area)"
DST_NAME  = "V2\n(receiving area)"
IN_CH     = 576    # source area channels
RANK      = 64     # bottleneck rank
OUT_CH    = 480    # receiving area channels
BU_KERNEL = 5      # receiving area's BU conv kernel size
BU_STRIDE = 1      # receiving area's BU conv stride

# ── layout ───────────────────────────────────────────────────────────────────
# Five stages placed along the x-axis
STAGE_X   = [0.0, 2.5, 4.5, 6.9, 9.4]  # centre x per stage
STAGE_W   = [1.6, 1.6, 1.4, 1.8, 1.6]  # box width per stage
MAX_H     = 3.5   # height for the largest channel count (IN_CH = 576)
MIN_H     = 0.5   # minimum drawn height (for rank = 64)

def ch_to_h(ch):
    """Scale channel count to a box height."""
    return MIN_H + (ch / IN_CH) * (MAX_H - MIN_H)

STAGE_H = [
    ch_to_h(IN_CH),   # source area
    ch_to_h(RANK),    # Conv2d 1×1 (compresses to rank)
    ch_to_h(RANK),    # ReLU (same channels)
    ch_to_h(OUT_CH),  # Conv2d k×k (expands to out_ch)
    ch_to_h(OUT_CH),  # receiving area
]

STAGE_LABELS = [
    SRC_NAME,
    f"Conv2d\n({IN_CH}→{RANK})",
    "ReLU",
    f"Conv2d\n({RANK}→{OUT_CH})",
    DST_NAME,
]

STAGE_SUBLABEL = [
    f"{IN_CH} channels",
    f"→ {RANK} channels",
    f"{RANK} channels",
    f"→ {OUT_CH} channels",
    f"{OUT_CH} channels",
]

# Colour per stage
C_AREA    = "#4A90D9"
C_BN_CONV = "#E8A838"
C_RELU    = "#E05C5C"
C_BU_CONV = "#5DBE6E"

STAGE_COLORS = [C_AREA, C_BN_CONV, C_RELU, C_BU_CONV, C_AREA]

# ── bracket labels ────────────────────────────────────────────────────────────
BN_BRACKET = (1, 2)   # stages 1-2 = bottleneck module
BU_BRACKET = (3, 3)   # stage 3 = receiving area's BU conv

# ── centering: compute symmetric x limits ────────────────────────────────────
X_MIN = STAGE_X[0]  - STAGE_W[0]  / 2
X_MAX = STAGE_X[-1] + STAGE_W[-1] / 2
X_PAD = 0.9   # equal padding on both sides

# ════════════════════════════════════════════════════════════════════════════
# DRAW
# ════════════════════════════════════════════════════════════════════════════
fig_w = (X_MAX - X_MIN) + 2 * X_PAD + 0.4
fig, ax = plt.subplots(figsize=(fig_w, 5.8))
ax.set_xlim(X_MIN - X_PAD, X_MAX + X_PAD)
ax.set_ylim(-2.0, MAX_H / 2 + 2.5)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_title(
    "How One BLT-VS Bottleneck Connection Works\n"
    rf"(example: V1 $\rightarrow$ V2, bottleneck rank = {RANK})",
    color="black", fontsize=12, fontweight="bold", pad=10,
)

# ── draw boxes ────────────────────────────────────────────────────────────────
box_tops = []
box_bots = []
for i, (x, w, h, color, label) in enumerate(
        zip(STAGE_X, STAGE_W, STAGE_H, STAGE_COLORS, STAGE_LABELS)):
    y_bot = -h / 2
    y_top =  h / 2
    box_tops.append(y_top)
    box_bots.append(y_bot)

    rect = FancyBboxPatch(
        (x - w / 2, y_bot), w, h,
        boxstyle="round,pad=0.10",
        facecolor=color, edgecolor="white", linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(rect)

    # Stage label inside box (only if box is tall enough)
    if h >= 0.7:
        ax.text(x, 0, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                zorder=4, multialignment="center")
    else:
        # Very narrow box: put label above
        ax.text(x, y_top + 0.15, label, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333", zorder=4,
                multialignment="center")

    # Channel sub-label below each box
    ax.text(x, y_bot - 0.25, STAGE_SUBLABEL[i],
            ha="center", va="top", fontsize=7.5, color="#555555", zorder=4)

# ── draw arrows between boxes ────────────────────────────────────────────────
ARROW_Y = 0   # arrows run through the vertical centre
for i in range(len(STAGE_X) - 1):
    x0 = STAGE_X[i]     + STAGE_W[i] / 2
    x1 = STAGE_X[i + 1] - STAGE_W[i + 1] / 2
    ax.annotate(
        "", xy=(x1, ARROW_Y), xytext=(x0, ARROW_Y),
        arrowprops=dict(
            arrowstyle="->,head_width=0.22,head_length=0.18",
            color="#333333", lw=1.8,
        ),
        zorder=5,
    )

# ── bracket: "Bottleneck module" spanning stages 1-2 ─────────────────────────
bk_x0 = STAGE_X[1] - STAGE_W[1] / 2 - 0.05
bk_x1 = STAGE_X[2] + STAGE_W[2] / 2 + 0.05
bk_y  = MAX_H / 2 + 0.5
bk_tick = 0.15
ax.plot([bk_x0, bk_x0, bk_x1, bk_x1],
        [bk_y - bk_tick, bk_y, bk_y, bk_y - bk_tick],
        color=C_BN_CONV, lw=1.5, zorder=3)
ax.text((bk_x0 + bk_x1) / 2, bk_y + 0.12,
        "Bottleneck module\n(compresses information)",
        ha="center", va="bottom", fontsize=8, color="#B07A10",
        fontweight="bold", multialignment="center")

# ── bracket: "Receiving area's BU conv" spanning stage 3 ─────────────────────
bu_x0 = STAGE_X[3] - STAGE_W[3] / 2 - 0.05
bu_x1 = STAGE_X[3] + STAGE_W[3] / 2 + 0.05
bu_y  = MAX_H / 2 + 0.5
ax.plot([bu_x0, bu_x0, bu_x1, bu_x1],
        [bu_y - bk_tick, bu_y, bu_y, bu_y - bk_tick],
        color=C_BU_CONV, lw=1.5, zorder=3)
ax.text((bu_x0 + bu_x1) / 2, bu_y + 0.12,
        "Receiving area's BU convolution",
        ha="center", va="bottom", fontsize=8, color="#2E8B47",
        fontweight="bold", multialignment="center")

# ── channel-count annotations on the arrows ───────────────────────────────────
channel_at_arrow = [IN_CH, RANK, RANK, OUT_CH]  # channels flowing on each arrow
for i, ch in enumerate(channel_at_arrow):
    x_mid = (STAGE_X[i] + STAGE_W[i] / 2 + STAGE_X[i + 1] - STAGE_W[i + 1] / 2) / 2
    ax.text(x_mid, ARROW_Y + 0.22, f"{ch} ch",
            ha="center", va="bottom", fontsize=7, color="#444444", zorder=5)

# ── legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_AREA,    edgecolor="white", label="Cortical area"),
    mpatches.Patch(facecolor=C_BN_CONV, edgecolor="white", label="Conv2d 1×1  (bottleneck compress)"),
    mpatches.Patch(facecolor=C_RELU,    edgecolor="white", label="ReLU activation"),
    mpatches.Patch(facecolor=C_BU_CONV, edgecolor="white", label="Conv2d  (BU conv of receiving area)"),
]
ax.legend(
    handles=legend_elements,
    loc="lower center", bbox_to_anchor=(0.5, -0.35),
    ncol=2,
    facecolor="#F5F5F5", edgecolor="#AAAAAA",
    labelcolor="#111111", fontsize=8,
    framealpha=0.95,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PATH}")
