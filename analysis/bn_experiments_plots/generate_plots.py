"""
Standalone plot generation from saved training metrics (.npz).
Reproduces all plots that train_net_copy_hooks.py creates after training.
No GPU, no model, no dataset needed — only the .npz file.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# ============================================================
# Configuration — SET THIS PATH
# ============================================================

LOSS_FILE = r"C:\Users\moehl\Logs\temp\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143\loss_blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143.npz"

OUT_DIR = os.path.dirname(LOSS_FILE)
PCA_FILE = os.path.join(OUT_DIR, "pca_results_streaming.npz")

TIMESTEPS = 12

# ============================================================
# Load data
# ============================================================

if not os.path.exists(LOSS_FILE):
    raise FileNotFoundError(f"Loss file not found: {LOSS_FILE}")

data = np.load(LOSS_FILE, allow_pickle=True)

train_losses = data["train_loss"]
val_losses = data["val_loss"]
train_accuracies = data["train_accuracies"]
val_accuracies = data["val_accuracies"]
val_accuracies_all = data["val_accuracies_all"] if "val_accuracies_all" in data.files else None

test_acc = data["test_accuracies"] if "test_accuracies" in data.files else None

epochs = np.arange(1, len(train_losses) + 1)

print(f"Loaded {len(epochs)} epochs from {LOSS_FILE}")
if test_acc is not None:
    print(f"Test accuracy: {test_acc}")

# ============================================================
# Compute best values
# ============================================================

best_val_acc = np.max(val_accuracies)
best_val_epoch = np.argmax(val_accuracies) + 1
train_acc_at_best = train_accuracies[best_val_epoch - 1]

best_val_loss = np.min(val_losses)
best_loss_epoch = np.argmin(val_losses) + 1
train_loss_at_best = train_losses[best_loss_epoch - 1]

# ============================================================
# 1. ACCURACY PLOT (Annotated)
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")

plt.scatter(best_val_epoch, best_val_acc, color="red", zorder=5)
plt.axvline(best_val_epoch, linestyle="--", alpha=0.5)

gap = train_acc_at_best - best_val_acc
plt.annotate(
    f"Best Val Acc: {best_val_acc:.2f}%\nEpoch {best_val_epoch}\nGap: {gap:.2f}%",
    (best_val_epoch, best_val_acc),
    textcoords="offset points",
    xytext=(-60, 20),
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve (Annotated)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accuracy_plot.png"), dpi=300)
plt.close()
print("  -> accuracy_plot.png")

# ============================================================
# 2. LOSS PLOT (Annotated)
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")

plt.scatter(best_loss_epoch, best_val_loss, color="green", zorder=5)
plt.axvline(best_loss_epoch, linestyle="--", alpha=0.5)

plt.annotate(
    f"Lowest Val Loss: {best_val_loss:.4f}\nEpoch {best_loss_epoch}",
    (best_loss_epoch, best_val_loss),
    textcoords="offset points",
    xytext=(-60, -25),
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Annotated)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_plot.png"), dpi=300)
plt.close()
print("  -> loss_plot.png")

# ============================================================
# 3. SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    {
        "Metric": [
            "Best Val Accuracy (%)",
            "Train Accuracy @ Best Val Epoch (%)",
            "Validation Loss @ Best Epoch",
            "Train Loss @ Best Val Epoch",
        ],
        "Value": [
            round(best_val_acc, 3),
            round(train_acc_at_best, 3),
            round(best_val_loss, 4),
            round(train_loss_at_best, 4),
        ],
    }
)

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis("off")
table = ax.table(cellText=summary.values, colLabels=summary.columns, loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_summary_table.png"), dpi=300)
plt.close()
print("  -> training_summary_table.png")

# ============================================================
# 4. TIMESTEP ACCURACY — Best Epoch
# ============================================================

if val_accuracies_all is not None and len(val_accuracies_all) > 0:
    best_epoch_idx = int(np.argmax(val_accuracies))
    best_epoch = best_epoch_idx + 1
    ts_acc = np.array(val_accuracies_all[best_epoch_idx], dtype=float)
    ts = np.arange(1, len(ts_acc) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(ts, ts_acc, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Validation Accuracy over Timesteps (Best Epoch {best_epoch})")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "val_accuracy_over_timesteps_best_epoch.png"), dpi=300)
    plt.close()
    print("  -> val_accuracy_over_timesteps_best_epoch.png")

# ============================================================
# 5. TIMESTEP ACCURACY — 5 evenly spaced epochs
# ============================================================

if val_accuracies_all is not None and len(val_accuracies_all) > 0:
    N = len(val_accuracies_all)

    if N >= 5:
        selected_epochs = np.linspace(1, N, 6)[1:]
        selected_epochs = np.unique(np.rint(selected_epochs).astype(int))
        selected_epochs = np.clip(selected_epochs, 1, N)
        selected_epochs = np.unique(selected_epochs)
        if len(selected_epochs) < 5:
            selected_epochs = np.unique(np.linspace(1, N, 5).astype(int))
    else:
        selected_epochs = np.arange(1, N + 1)

    plt.figure(figsize=(8, 5))
    for ep in selected_epochs:
        ts_acc = np.array(val_accuracies_all[ep - 1], dtype=float)
        ts = np.arange(1, len(ts_acc) + 1)
        plt.plot(ts, ts_acc, marker="o", label=f"Epoch {ep}")
    plt.xlabel("Timestep")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy over Timesteps (5 checkpoints)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "val_accuracy_over_timesteps_5epochs.png"), dpi=300)
    plt.close()
    print("  -> val_accuracy_over_timesteps_5epochs.png")

# ============================================================
# 6. RECURRENCE ANALYSIS — Timestep curve (final epoch)
# ============================================================

if val_accuracies_all is not None and len(val_accuracies_all) > 0:
    val_all = np.array(val_accuracies_all, dtype=float)
    n_epochs, n_timesteps = val_all.shape
    last_epoch = val_all[-1]

    t1 = last_epoch[0]
    tmax = last_epoch.max()
    rec_score = tmax - t1
    percent_gain = (rec_score / t1) * 100 if t1 != 0 else 0

    print(f"\nRecurrence Analysis:")
    print(f"  Final Epoch: {n_epochs}")
    print(f"  T1 Accuracy: {t1:.2f}")
    print(f"  Tmax Accuracy: {tmax:.2f}")
    print(f"  Recurrence Score (delta): {rec_score:.2f}")
    print(f"  Relative Gain: {percent_gain:.2f}%")

    plt.figure()
    plt.plot(range(1, n_timesteps + 1), last_epoch, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy over Timesteps (Final Epoch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timestep_curve_last_epoch.png"), dpi=300)
    plt.close()
    print("  -> timestep_curve_last_epoch.png")

    # ============================================================
    # 7. RECURRENCE ANALYSIS — Timestep table
    # ============================================================

    columns = [f"t{i+1}" for i in range(n_timesteps)] + ["t_max", "delta (tmax-t1)", "delta (%)"]
    rows = []
    for e in range(n_epochs):
        row = val_all[e]
        t1_e = row[0]
        tmax_e = row.max()
        delta = tmax_e - t1_e
        pct = (delta / t1_e) * 100 if t1_e != 0 else 0
        full_row = list(np.round(row, 2)) + [round(tmax_e, 2), round(delta, 2), round(pct, 2)]
        rows.append(full_row)

    fig, ax = plt.subplots(figsize=(14, 0.4 * n_epochs + 2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        rowLabels=[f"E{e+1}" for e in range(n_epochs)],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title("Validation Accuracy - All Epochs (Timestep Summary)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timestep_table.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> timestep_table.png")

    # ============================================================
    # 8. RECURRENCE GAIN HEATMAP
    # ============================================================

    gain = val_all - val_all[:, 0:1]

    plt.figure(figsize=(10, 6))
    im = plt.imshow(gain, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Gain relative to t1 (Accuracy %)")
    plt.xlabel("Timestep")
    plt.ylabel("Epoch")
    plt.title("Recurrence Gain over Training (Relative to t1)")
    plt.xticks(range(n_timesteps), [f"t{i+1}" for i in range(n_timesteps)])
    plt.yticks(range(0, n_epochs, max(1, n_epochs // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recurrence_gain_heatmap.png"), dpi=300)
    plt.close()
    print("  -> recurrence_gain_heatmap.png")

# ============================================================
# 9. PCA DIMENSIONALITY PLOTS (if pca_results_streaming.npz exists)
# ============================================================

if os.path.exists(PCA_FILE):
    pca_data = np.load(PCA_FILE)

    areas = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

    total_channels = {
        "Retina": 32,
        "LGN": 32,
        "V1": 576,
        "V2": 480,
        "V3": 352,
        "V4": 256,
        "LOC": 352,
    }

    levels = [90, 95, 99]

    for level in levels:
        dim_matrix = []
        for area in areas:
            row = []
            for t in range(TIMESTEPS):
                key = f"{area}_t{t}_channels_{level}"
                row.append(pca_data[key][0] if key in pca_data else 0)
            row.append(total_channels[area])
            dim_matrix.append(row)

        dim_matrix = np.array(dim_matrix)
        heatmap_abs = dim_matrix[:, :-1]
        totals = np.array([total_channels[a] for a in areas])[:, None]
        heatmap_rel = heatmap_abs / totals

        fig, axes = plt.subplots(
            2, 2,
            figsize=(22, 10),
            gridspec_kw={"height_ratios": [1, 0.65], "wspace": 0.35, "hspace": 0.12},
        )

        # Absolute heatmap
        ax = axes[0, 0]
        im = ax.imshow(heatmap_abs, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Channels for {level}% variance")
        ax.set_xticks(range(TIMESTEPS))
        ax.set_xticklabels(range(TIMESTEPS))
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Visual Area")
        ax.set_title(f"Representation Dimensionality ({level}% variance)")

        # Relative heatmap
        ax = axes[0, 1]
        im = ax.imshow(heatmap_rel, aspect="auto", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Fraction of total channels")
        ax.set_xticks(range(TIMESTEPS))
        ax.set_xticklabels(range(TIMESTEPS))
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Visual Area")
        ax.set_title(f"Relative Dimensionality ({level}% variance)")

        # Absolute table
        ax = axes[1, 0]
        ax.axis("off")
        tbl = ax.table(
            cellText=dim_matrix,
            rowLabels=areas,
            colLabels=[f"t{i}" for i in range(TIMESTEPS)] + ["Total"],
            cellLoc="center",
            bbox=[0, 0.20, 1, 0.75],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.6)

        # Relative table
        ax = axes[1, 1]
        ax.axis("off")
        rel_matrix = np.round(heatmap_rel * 100, 1)
        rel_matrix = np.concatenate([rel_matrix, np.full((len(areas), 1), 100)], axis=1)
        tbl = ax.table(
            cellText=rel_matrix,
            rowLabels=areas,
            colLabels=[f"t{i}" for i in range(TIMESTEPS)] + ["Total"],
            cellLoc="center",
            bbox=[0, 0.20, 1, 0.75],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.6)

        plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05)
        plt.savefig(os.path.join(OUT_DIR, f"pca_dimensionality_{level}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> pca_dimensionality_{level}.png")

    print("PCA plots saved.")
else:
    print(f"PCA file not found ({PCA_FILE}), skipping PCA plots.")

print("\nDone! All plots saved to:", OUT_DIR)
