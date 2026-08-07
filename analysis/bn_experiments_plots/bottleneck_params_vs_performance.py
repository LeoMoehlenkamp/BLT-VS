"""
Scatter plot: bottleneck parameter count (x) vs. best validation accuracy (y).

X-axis  – total parameters in the bottleneck Conv2d layer(s) (log scale).
          Each bottleneck edge uses Conv2d(in_ch, rank, kernel_size=1):
              params_per_edge = in_channels × rank + rank  (weight + bias)
          Higher value  →  looser bottleneck  →  more information can flow.

Y-axis  – best validation accuracy (%) read from the LaTeX performance table.

Colour  – connectivity family (BU / BU-Skip / BU-TD / BU-TD-Skip).
Marker  – bottleneck scope (within the BU family only).
Dotted horizontal lines show each family's no-bottleneck baseline.

Data source: hardcoded from the LaTeX performance summary table.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# ARCHITECTURE CONSTANTS
# ============================================================

# Output channels per area (from blt_vs_bottleneck_modular.py)
CHANNEL_SIZES = {
    "Retina": 32,
    "LGN":    32,
    "V1":     576,
    "V2":     480,
    "V3":     352,
    "V4":     256,
    "LOC":    352,
}


def bn_params(connections: dict) -> int:
    """Total parameters in a set of bottleneck Conv2d(in_ch, rank, 1) layers.
    Strips '_skip' / '_td' suffix to resolve the source area's channel count.
    """
    total = 0
    for edge, rank in connections.items():
        base_edge = edge.replace("_skip", "").replace("_td", "")
        src = base_edge.split("->")[0]
        in_ch = CHANNEL_SIZES[src]
        total += in_ch * rank + rank   # weight + bias
    return total


# ============================================================
# BOTTLENECK EDGE SETS  (one function per group)
# ============================================================

def cfg_v1v2(r):
    return {"V1->V2": r}

def cfg_v2v3(r):
    return {"V2->V3": r}

def cfg_both(r):
    return {"V1->V2": r, "V2->V3": r}

def cfg_bu_bnall(r):
    # BU model: all feedforward connections incl. Retina→LGN and LGN→V1
    return {
        "Retina->LGN": r, "LGN->V1": r,
        "V1->V2": r, "V2->V3": r, "V3->V4": r, "V4->LOC": r,
    }

def cfg_skip_bnall(r):
    # BU-Skip model: feedforward + skip connections
    return {
        "V1->V2": r, "V2->V3": r, "V3->V4": r, "V4->LOC": r,
        "V1->V4_skip": r, "V4->V1_skip": r,
    }

def cfg_td_bnall(r):
    # BU-TD model: feedforward + top-down connections
    return {
        "V1->V2": r, "V2->V3": r, "V3->V4": r, "V4->LOC": r,
        "V1->LGN_td": r, "V2->V1_td": r, "V3->V2_td": r,
        "V4->V3_td": r, "LOC->V4_td": r,
    }

def cfg_tdskip_bnall(r):
    # BU-TD-Skip model: feedforward + top-down + skip connections
    return {
        "V1->V2": r, "V2->V3": r, "V3->V4": r, "V4->LOC": r,
        "V1->LGN_td": r, "V2->V1_td": r, "V3->V2_td": r,
        "V4->V3_td": r, "LOC->V4_td": r,
        "V1->V4_skip": r, "V4->V1_skip": r,
    }


# ============================================================
# PERFORMANCE DATA  (from LaTeX table, best val accuracy %)
# ============================================================
# Each list: (rank, best_val_accuracy_pct)

BU_V1V2 = [
    (360, 75.60), (192, 74.35), (128, 74.70), (96, 74.16), (64, 74.96),
    (32, 73.93), (16, 73.61), (12, 72.96), (8, 71.29), (6, 70.98), (4, 69.39),
]
BU_V2V3 = [
    (256, 75.23), (128, 73.58), (64, 74.08), (32, 72.90),
    (16, 72.99), (12, 72.40), (8, 71.38), (6, 68.86),
]
BU_BOTH = [
    (32, 72.63), (24, 72.33), (12, 69.61),
]
BU_BNALL = [
    (96, 72.26), (64, 72.18), (32, 69.20), (16, 65.79),
]

SKIP_BNALL = [
    (96, 72.12), (64, 71.25), (32, 68.36), (16, 69.20),
]
TD_BNALL = [
    (96, 74.26), (64, 73.52), (32, 71.57), (16, 67.15),
]
TDSKIP_BNALL = [
    (96, 74.59), (64, 73.83), (32, 71.98), (16, 70.20),
]

BASELINES = {
    "BU":           75.08,
    "BU-Skip":      73.79,
    "BU-TD":        77.01,
    "BU-TD-Skip":   74.98,
}

# ============================================================
# PLOT SETTINGS
# ============================================================

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\bottleneck_params_vs_performance.png"

FAMILY_COLORS = {
    "BU":           "#1565C0",   # deep blue
    "BU-Skip":      "#2E7D32",   # deep green
    "BU-TD":        "#E65100",   # deep orange
    "BU-TD-Skip":   "#AD1457",   # deep pink/red
}

# Markers distinguish bottleneck scope within the BU family
BU_MARKERS = {
    "V1V2":         "o",
    "V2V3":         "s",
    "V1V2+V2V3":    "D",
    "bnall":        "^",
}

# ============================================================
# HELPER: plot one group of (rank, perf) pairs
# ============================================================

def plot_group(ax, ranks_perfs, cfg_fn, color, marker, label,
               ms=70, line_alpha=0.4, annotate_ranks=False):
    xs = [bn_params(cfg_fn(r)) for r, _ in ranks_perfs]
    ys = [perf for _, perf in ranks_perfs]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    ranks_sorted = [ranks_perfs[i][0] for i in order]

    ax.plot(xs, ys, color=color, alpha=line_alpha, linewidth=1.4, zorder=2)
    sc = ax.scatter(xs, ys, s=ms, color=color, marker=marker,
                    edgecolor="white", linewidth=0.7, zorder=3, label=label)

    if annotate_ranks:
        for x, y, r in zip(xs, ys, ranks_sorted):
            ax.annotate(str(r), (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=6.5, color=color, alpha=0.85)
    return sc


# ============================================================
# DRAW
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

# ── Baseline horizontal dotted reference lines ────────────────────────────────
for family, perf in BASELINES.items():
    ax.axhline(perf, color=FAMILY_COLORS[family], linewidth=1.3,
               linestyle=":", alpha=0.65, zorder=1)

# ── BU family (multiple bottleneck scopes) ────────────────────────────────────
c_bu = FAMILY_COLORS["BU"]
plot_group(ax, BU_V1V2,  cfg_v1v2,    c_bu, BU_MARKERS["V1V2"],
           label="BU – V1→V2",       annotate_ranks=True)
plot_group(ax, BU_V2V3,  cfg_v2v3,    c_bu, BU_MARKERS["V2V3"],
           label="BU – V2→V3",       annotate_ranks=True)
plot_group(ax, BU_BOTH,  cfg_both,    c_bu, BU_MARKERS["V1V2+V2V3"],
           label="BU – V1V2 + V2V3", annotate_ranks=True)
plot_group(ax, BU_BNALL, cfg_bu_bnall, c_bu, BU_MARKERS["bnall"],
           label="BU – all BU paths", annotate_ranks=True)

# ── Other families (bnall only) ────────────────────────────────────────────────
plot_group(ax, SKIP_BNALL,   cfg_skip_bnall,   FAMILY_COLORS["BU-Skip"],
           "^", "BU-Skip – bnall",   annotate_ranks=True)
plot_group(ax, TD_BNALL,     cfg_td_bnall,     FAMILY_COLORS["BU-TD"],
           "^", "BU-TD – bnall",     annotate_ranks=True)
plot_group(ax, TDSKIP_BNALL, cfg_tdskip_bnall, FAMILY_COLORS["BU-TD-Skip"],
           "^", "BU-TD-Skip – bnall", annotate_ranks=True)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlabel("Bottleneck parameter count (log scale)\n"
              "← tighter bottleneck  ·  looser bottleneck →",
              fontsize=11)
ax.set_ylabel("Best validation accuracy (%)", fontsize=11)
ax.set_title("Bottleneck strength vs. classification performance\n"
             "(each point = one trained model; rank annotated; dotted lines = no-bottleneck baseline)",
             fontsize=12, fontweight="bold")
ax.grid(True, which="both", alpha=0.25)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
# Scatter / line groups (already labelled above)
scatter_handles, scatter_labels = ax.get_legend_handles_labels()

# Baseline reference lines
baseline_handles = [
    Line2D([0], [0], color=FAMILY_COLORS[f], linewidth=1.3, linestyle=":",
           alpha=0.85, label=f"{f} baseline ({v:.2f}%)")
    for f, v in BASELINES.items()
]

ax.legend(
    handles=scatter_handles + baseline_handles,
    labels=scatter_labels + [h.get_label() for h in baseline_handles],
    fontsize=8.5, loc="lower right", ncol=2,
    title="Model group", title_fontsize=9,
    framealpha=0.9,
)

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
