import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\moehl\Logs\Exp\final"

RUN_1 = r"C:\Users\moehl\Logs\Exp\final\BNV1V2\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846"
RUN_2 = r"C:\Users\moehl\Logs\Exp\final\BNall\blt_vs_bottleneck__miniecoset__ts12__bn-RetinaLGN-16_LGNV1-16_V1V2-16_V2V3-16_V3V4-16_V4LOC-16__20260402_123451"

LABEL_1 = "No BN"
LABEL_2 = "BN all(16)"

FIGURE_TITLE = "Comparison of Two Runs"
SAVE_PATH = os.path.join(BASE_DIR, "comparison_two_runs.png")

# only the plots you want to include
PLOTS = [
    ("accuracy_plot.png", "Accuracy"),
    ("recurrence_gain_summary.png", "Rec Gain"),
    ("timestep_table.png", "Timestep Table"),
    ("pca_dimensionality_95.png", "PCA Dim"),
]

# =========================
# PREPARE RUNS
# =========================

RUNS = {
    LABEL_1: RUN_1,
    LABEL_2: RUN_2,
}

# =========================
# CREATE FIGURE
# =========================

n_rows = len(PLOTS)
n_cols = len(RUNS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

# handle 1 row / 1 col cases safely
if n_rows == 1 and n_cols == 1:
    axes = [[axes]]
elif n_rows == 1:
    axes = [axes]
elif n_cols == 1:
    axes = [[ax] for ax in axes]

for col, (label, folder) in enumerate(RUNS.items()):
    folder_path = os.path.join(BASE_DIR, folder)

    for row, (plot_file, row_title) in enumerate(PLOTS):
        ax = axes[row][col]

        img_path = os.path.join(folder_path, plot_file)

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing\n{plot_file}", ha="center", va="center", fontsize=14)

        ax.axis("off")

        # column titles
        if row == 0:
            ax.set_title(label, fontsize=16)

        # row labels
        if col == 0:
            ax.set_ylabel(row_title, fontsize=14)

fig.suptitle(FIGURE_TITLE, fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("Comparison figure saved to:", SAVE_PATH)