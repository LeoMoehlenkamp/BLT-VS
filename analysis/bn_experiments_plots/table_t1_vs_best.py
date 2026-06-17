"""
Generate a table of t=1 vs best validation accuracy for multiple models.
Columns: Model, t=1 Acc (%), t_max Acc (%), Gap (Δ%)

Configure MODELS below: list of (npz_path, label) tuples.
"""

import numpy as np
import os

# ============================================================
# CONFIGURE HERE — add/remove models as needed
# ============================================================

MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNnone_BU_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523.npz", "BNnone_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNall_BU_Skip\BNall64_BU_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-bnall64skip__20260416_130242\loss_blt_vs_bottleneck__miniecoset__ts12__bn-bnall64skip__20260416_130242.npz", "BNall64_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNnone_BU_TD\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158\loss_blt_vs_bottleneck__miniecoset__ts12__bnall96_BU-TD__20260421_120158.npz", "BNnone_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNall_BU_TD\BNall64_BU_TD\blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-TD__20260422_112005\loss_blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-TD__20260422_112005.npz", "BNall64_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNnone_BU_TD_Skip\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019.npz", "BNnone_BU_ TD_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNall_BU_TD_Skip\BNall32_BU_TD_Skip\blt_vs_bottleneck__miniecoset__ts12__bnall32_BU-TD-Skip__20260602_005408\loss_blt_vs_bottleneck__miniecoset__ts12__bnall32_BU-TD-Skip__20260602_005408.npz", "BNall32_BU_ TD_Skip"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\t1_vs_best_table.png"

# ============================================================
# LOAD DATA
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []

for npz_path, label in MODELS:
    if not os.path.exists(npz_path):
        print(f"WARNING: File not found, skipping: {npz_path}")
        continue

    data = np.load(npz_path)

    if "val_accuracies_all" not in data:
        print(f"WARNING: 'val_accuracies_all' not found in {npz_path}, skipping")
        continue

    val_all = data["val_accuracies_all"]  # shape: (epochs, timesteps)
    t1_acc = val_all[-1, 0]              # final epoch, t=1
    best_acc = val_all.max()              # best across all epochs & timesteps
    gap = best_acc - t1_acc

    rows.append((label, f"{t1_acc:.2f}", f"{best_acc:.2f}", f"{gap:.2f}"))

if len(rows) == 0:
    print("ERROR: No valid models loaded.")
    exit(1)

# ============================================================
# RENDER TABLE
# ============================================================

col_labels = ["Model", "t=1 Acc (%)", "t_max Acc (%)", "Δ (%)"]

fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(rows)))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.6)

# Style header row
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row shading
for i in range(1, len(rows) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor("#D9E2F3")
        else:
            cell.set_facecolor("#FFFFFF")

fig.suptitle("First-Timestep vs. Best Accuracy", fontsize=13, fontweight="bold", y=0.95)
plt.tight_layout()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()

# Also print to console
print()
print(f"{'Model':<20} {'t=1 Acc (%)':>12} {'t_max Acc (%)':>14} {'Δ (%)':>8}")
print("-" * 58)
for label, t1, best, gap in rows:
    print(f"{label:<20} {t1:>12} {best:>14} {gap:>8}")
