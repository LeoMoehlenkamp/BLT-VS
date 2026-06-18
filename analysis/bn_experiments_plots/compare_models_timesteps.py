"""
Compare validation accuracy over timesteps across multiple models.

Configure the MODELS list below, then just run:
    python analysis/bn_experiments_plots/compare_models_timesteps.py
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ============================================================
# CONFIGURE HERE
# ============================================================

# Each entry: (path_to_log_dir, display_name)
MODELS = [
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143", "BNnone_BU_ TD_Skip_Ecoset"),
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731", "BNall32_BU_TD_Skip_Ecoset"),
]

OUTPUT_PATH = r"C:\Users\moehl\Logs\Plots_BA\comparison_best_epoch_Ecoset"
OUTPUT_PATH_GAIN = r"C:\Users\moehl\Logs\Plots_BA\comparison_best_epoch_normalized_Ecoset"
USE_LAST_EPOCH = False  # True = last epoch, False = best epoch

COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#9b5de5", "#f15bb5"]


def find_npz(log_dir):
    """Find the loss_*.npz file in a log directory."""
    matches = glob.glob(os.path.join(log_dir, "loss_*.npz"))
    if not matches:
        print(f"WARNING: No loss_*.npz found in {log_dir}, skipping.")
        return None
    return matches[0]


def load_model_data(npz_path, use_last=False):
    """Load timestep accuracies from npz and return (ts_acc, epoch_label)."""
    data = np.load(npz_path)

    if "val_accuracies_all" not in data.files:
        print(f"WARNING: 'val_accuracies_all' not in {npz_path}, skipping.")
        return None, None

    val_all = data["val_accuracies_all"]  # shape: (epochs, timesteps)
    val_acc = data["val_accuracies"]       # shape: (epochs,)

    if use_last:
        idx = len(val_acc) - 1
    else:
        idx = int(np.argmax(val_acc))

    ts_acc = np.array(val_all[idx], dtype=float)
    epoch = idx + 1
    mean_acc = float(np.mean(ts_acc))

    return ts_acc, epoch, mean_acc


def derive_name(log_dir):
    """Derive a short model name from the directory path."""
    base = os.path.basename(os.path.normpath(log_dir))
    # Strip trailing timestamp (e.g. __20260316_210800)
    parts = base.rsplit("__", 1)
    if len(parts) == 2 and len(parts[1]) >= 8 and parts[1][:8].isdigit():
        return parts[0]
    return base


def main():
    # Collect data for both plots
    model_curves = []
    for log_dir, name in MODELS:
        npz_path = find_npz(log_dir)
        if npz_path is None:
            continue

        result = load_model_data(npz_path, use_last=USE_LAST_EPOCH)
        if result[0] is None:
            continue

        ts_acc, epoch, mean_acc = result
        model_curves.append((name, ts_acc, epoch, mean_acc))

    if not model_curves:
        print("ERROR: No valid models found. Check your MODELS paths.")
        sys.exit(1)

    # --- Plot 1: Absolute accuracy ---
    plt.figure(figsize=(9, 5))
    for i, (name, ts_acc, epoch, mean_acc) in enumerate(model_curves):
        timesteps = np.arange(1, len(ts_acc) + 1)
        label = f"{name} (ep{epoch}, mean={mean_acc:.1f}%)"
        plt.plot(timesteps, ts_acc, marker="o", label=label,
                 color=COLORS[i % len(COLORS)])

    plt.xlabel("Timestep")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy over Timesteps — Model Comparison")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {OUTPUT_PATH}")

    # --- Plot 2: Recurrence gain (normalized to t1=0) ---
    plt.figure(figsize=(9, 5))
    for i, (name, ts_acc, epoch, mean_acc) in enumerate(model_curves):
        timesteps = np.arange(1, len(ts_acc) + 1)
        gain = ts_acc - ts_acc[0]
        total_gain = gain[-1]
        label = f"{name} (t1={ts_acc[0]:.1f}%, +{total_gain:.1f}pp)"
        plt.plot(timesteps, gain, marker="o", label=label,
                 color=COLORS[i % len(COLORS)])

    plt.xlabel("Timestep")
    plt.ylabel("Accuracy gain over t1 (pp)")
    plt.title("Recurrence Gain — Model Comparison (all aligned at t1=0)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH_GAIN, dpi=300)
    plt.close()
    print(f"Saved gain plot to {OUTPUT_PATH_GAIN}")


if __name__ == "__main__":
    main()
