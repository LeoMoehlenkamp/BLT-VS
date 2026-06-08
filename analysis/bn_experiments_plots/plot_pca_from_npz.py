"""
Recreate PCA dimensionality plots from existing pca_results_streaming.npz files.

Configure MODELS below, then run:
    python analysis/bn_experiments_plots/plot_pca_from_npz.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURE HERE
# ============================================================

# Each entry: (path_to_pca_results_streaming.npz, display_label)
MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800\pca_results_streaming.npz", "BN-none"),
    # Add more models here:
    # (r"C:\...\pca_results_streaming.npz", "BN-12"),
]

SAVE_DIR = r"C:\Users\moehl\Logs\Plots_BA\pca"

# Which variance thresholds to plot (90, 95, 99)
LEVELS = [90, 95, 99]

N_TIMESTEPS = 12  # adjust if your model uses a different number

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

# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_pca(npz_path, label, save_dir, levels, n_timesteps):
    if not os.path.exists(npz_path):
        print(f"WARNING: File not found, skipping: {npz_path}")
        return

    data = np.load(npz_path)
    os.makedirs(save_dir, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")

    for level in levels:
        dim_matrix = []
        for area in AREAS:
            row = []
            for t in range(n_timesteps):
                key = f"{area}_t{t}_channels_{level}"
                row.append(int(data[key][0]) if key in data else 0)
            row.append(TOTAL_CHANNELS[area])
            dim_matrix.append(row)

        dim_matrix = np.array(dim_matrix)
        heatmap_abs = dim_matrix[:, :-1]
        totals = np.array([TOTAL_CHANNELS[a] for a in AREAS])[:, None]
        heatmap_rel = heatmap_abs / totals

        fig, axes = plt.subplots(
            2, 2,
            figsize=(22, 10),
            gridspec_kw={"height_ratios": [1, 0.65], "wspace": 0.35, "hspace": 0.12},
        )

        fig.suptitle(f"{label} — PCA Dimensionality ({level}% variance)", fontsize=13)

        # Absolute heatmap
        ax = axes[0, 0]
        im = ax.imshow(heatmap_abs, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Channels for {level}% variance")
        ax.set_xticks(range(n_timesteps))
        ax.set_xticklabels(range(n_timesteps))
        ax.set_yticks(range(len(AREAS)))
        ax.set_yticklabels(AREAS)
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
        ax.set_yticks(range(len(AREAS)))
        ax.set_yticklabels(AREAS)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Visual Area")
        ax.set_title(f"Relative Dimensionality ({level}% variance)")

        # Absolute table
        ax = axes[1, 0]
        ax.axis("off")
        table_abs = ax.table(
            cellText=dim_matrix,
            rowLabels=AREAS,
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
        rel_matrix = np.concatenate([rel_matrix, np.full((len(AREAS), 1), 100.0)], axis=1)
        table_rel = ax.table(
            cellText=rel_matrix,
            rowLabels=AREAS,
            colLabels=[f"t{i}" for i in range(n_timesteps)] + ["Total"],
            cellLoc="center",
            bbox=[0, 0.20, 1, 0.75],
        )
        table_rel.auto_set_font_size(False)
        table_rel.set_fontsize(11)
        table_rel.scale(1.2, 1.6)

        plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05)

        save_path = os.path.join(save_dir, f"{safe_label}_pca_{level}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

for npz_path, label in MODELS:
    print(f"\nProcessing: {label}")
    plot_pca(npz_path, label, SAVE_DIR, LEVELS, N_TIMESTEPS)

print("\nDone.")
