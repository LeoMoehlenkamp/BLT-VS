import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\moehl\Logs\Exp\final\BNV1V2"
SAVE_PREFIX = "V1V2"

NO_BN_NPZ_PATH = r"C:\Users\moehl\Logs\Exp\final\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800\loss_blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800.npz"

SHOW_GAP_LINES = True

# =========================
# HELPERS
# =========================

def get_bn_value(folder_name):
    """
    Extract bottleneck size from folder name.
    Examples:
        bn-none -> 0
        bn-V2V3-144 -> 144
        bn-V1V2-96 -> 96
    """
    if "bn-none" in folder_name:
        return 0

    match = re.search(r"bn-[A-Za-z0-9]+-(\d+)", folder_name)
    if match:
        return int(match.group(1))

    return None


def extract_metrics_from_npz(npz_path):
    data = np.load(npz_path)

    if "val_accuracies_all" not in data:
        raise ValueError(f"'val_accuracies_all' not found in {npz_path}")

    val = data["val_accuracies_all"]   # shape: [epochs, timesteps]

    t1_all_epochs = val[:, 0]
    tmax_all_epochs = val.max(axis=1)

    delta_abs = tmax_all_epochs - t1_all_epochs

    with np.errstate(divide='ignore', invalid='ignore'):
        delta_pct = np.where(
            tmax_all_epochs != 0,
            (delta_abs / tmax_all_epochs) * 100,
            np.nan
        )

    if "val_accuracies" in data:
        val_acc = data["val_accuracies"]
        best_epoch = np.argmax(val_acc)
        best_val = val_acc[best_epoch]
        best_t1 = val[best_epoch, 0]
    else:
        best_val = np.nan
        best_t1 = np.nan

    return {
        "avg_abs": np.nanmean(delta_abs),
        "max_abs": np.nanmax(delta_abs),
        "avg_pct": np.nanmean(delta_pct),
        "max_pct": np.nanmax(delta_pct),
        "best_val": best_val,
        "best_t1": best_t1,
    }


# =========================
# COLLECT DATA
# =========================

bn_values = []
avg_abs_list = []
max_abs_list = []
avg_pct_list = []
max_pct_list = []
best_val_list = []
best_t1_list = []

# ---- 1) Add No-BN run manually ----
if os.path.exists(NO_BN_NPZ_PATH):
    try:
        metrics = extract_metrics_from_npz(NO_BN_NPZ_PATH)
        bn_values.append(0)
        avg_abs_list.append(metrics["avg_abs"])
        max_abs_list.append(metrics["max_abs"])
        avg_pct_list.append(metrics["avg_pct"])
        max_pct_list.append(metrics["max_pct"])
        best_val_list.append(metrics["best_val"])
        best_t1_list.append(metrics["best_t1"])
        print("Added No-BN run.")
    except Exception as e:
        print(f"Could not load No-BN run: {e}")
else:
    print("No-BN npz path does not exist:", NO_BN_NPZ_PATH)

# ---- 2) Add all bottleneck runs from BASE_DIR ----
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    bn = get_bn_value(folder)
    if bn is None:
        continue

    # skip bn-none inside BASE_DIR to avoid accidental duplicates
    if bn == 0:
        continue

    npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
    if not npz_files:
        print(f"No npz in {folder}")
        continue

    npz_path = npz_files[0]

    try:
        metrics = extract_metrics_from_npz(npz_path)
    except Exception as e:
        print(f"Skipping {folder}: {e}")
        continue

    bn_values.append(bn)
    avg_abs_list.append(metrics["avg_abs"])
    max_abs_list.append(metrics["max_abs"])
    avg_pct_list.append(metrics["avg_pct"])
    max_pct_list.append(metrics["max_pct"])
    best_val_list.append(metrics["best_val"])
    best_t1_list.append(metrics["best_t1"])

# =========================
# CHECK IF ANYTHING WAS FOUND
# =========================

if len(bn_values) == 0:
    raise ValueError("No valid experiment data found.")

# =========================
# SORT (No BN first, then big -> small)
# =========================

combined = list(zip(
    bn_values,
    avg_abs_list,
    max_abs_list,
    avg_pct_list,
    max_pct_list,
    best_val_list,
    best_t1_list
))

no_bn = [x for x in combined if x[0] == 0]
others = [x for x in combined if x[0] != 0]

others_sorted = sorted(others, key=lambda x: x[0], reverse=True)
final = no_bn + others_sorted

(
    bn_values,
    avg_abs_list,
    max_abs_list,
    avg_pct_list,
    max_pct_list,
    best_val_list,
    best_t1_list
) = zip(*final)

bn_values = list(bn_values)
avg_abs_list = list(avg_abs_list)
max_abs_list = list(max_abs_list)
avg_pct_list = list(avg_pct_list)
max_pct_list = list(max_pct_list)
best_val_list = list(best_val_list)
best_t1_list = list(best_t1_list)

# =========================
# COMMON X (categorical)
# =========================

x = np.arange(len(bn_values))
labels = ["No BN" if v == 0 else str(v) for v in bn_values]

# =========================
# PLOT 1: RECURRENCE SUMMARY
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, avg_abs_list, marker='o')
axes[0, 0].set_title("Avg Recurrence Gain (abs)")
axes[0, 0].set_ylabel("Δ Accuracy")

axes[0, 1].plot(x, max_abs_list, marker='o')
axes[0, 1].set_title("Max Recurrence Gain (abs)")

axes[1, 0].plot(x, avg_pct_list, marker='o')
axes[1, 0].set_title("Avg Recurrence Gain (%)")
axes[1, 0].set_ylabel("%")

axes[1, 1].plot(x, max_pct_list, marker='o')
axes[1, 1].set_title("Max Recurrence Gain (%)")

for ax in axes.flat:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Bottleneck Size")
    ax.grid(True)

plt.suptitle(f"Bottleneck Summary ({SAVE_PREFIX})")
plt.tight_layout()

save_path_summary = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_bottleneck_summary.png")
plt.savefig(save_path_summary, dpi=300, bbox_inches="tight")
plt.close()

print("Saved recurrence plot to:", save_path_summary)

# =========================
# PLOT 2: BEST ACCURACY + T1 + GAP LABELS
# =========================

plt.figure(figsize=(10, 6))

plt.plot(x, best_val_list, marker='o', label="Best validation accuracy")
plt.plot(x, best_t1_list, marker='s', label="T1 accuracy at best epoch")

if SHOW_GAP_LINES:
    for i, (best_val, best_t1) in enumerate(zip(best_val_list, best_t1_list)):
        if not (np.isnan(best_val) or np.isnan(best_t1)):
            y_low = min(best_val, best_t1)
            y_high = max(best_val, best_t1)
            plt.vlines(i, y_low, y_high, linestyles='dashed', alpha=0.5)

for i, v in enumerate(best_val_list):
    if not np.isnan(v):
        plt.text(i, v + 0.05, f"{v:.1f}", ha='center', va='bottom')

for i, v in enumerate(best_t1_list):
    if not np.isnan(v):
        plt.text(i, v - 0.05, f"{v:.1f}", ha='center', va='top')

for i, (best_val, best_t1) in enumerate(zip(best_val_list, best_t1_list)):
    if np.isnan(best_val) or np.isnan(best_t1):
        continue

    gap = best_val - best_t1
    y_mid = (best_val + best_t1) / 2

    plt.text(
        i + 0.08,
        y_mid,
        f"Δ {gap:.1f}",
        ha='left',
        va='center',
        fontsize=10
    )

plt.xticks(x, labels, rotation=45)
plt.xlabel("Bottleneck Size")
plt.ylabel("Accuracy (%)")
plt.title(f"Best Accuracy vs T1 at Best Epoch ({SAVE_PREFIX})")
plt.grid(True)
plt.legend()

plt.tight_layout()

save_path_acc = os.path.join(BASE_DIR, f"{SAVE_PREFIX}_bottleneck_accuracy_vs_t1.png")
plt.savefig(save_path_acc, dpi=300, bbox_inches="tight")
plt.close()

print("Saved accuracy plot to:", save_path_acc)

print("\nDONE.")