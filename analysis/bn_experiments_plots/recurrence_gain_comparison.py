"""
Recurrence Gain comparison across multiple models.
Grouped bar chart: each model gets bars for Avg Δ (abs) and Avg Δ (%).

Configure MODELS below: list of (npz_path, label) tuples.
"""

import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def find_npz(log_dir):
    """Return the first loss_*.npz found in log_dir, or None."""
    if os.path.isfile(log_dir):
        return log_dir
    matches = glob.glob(os.path.join(log_dir, "loss_*.npz"))
    if not matches:
        print(f"WARNING: No loss_*.npz found in {log_dir}, skipping.")
        return None
    return matches[0]

# ============================================================
# CONFIGURE HERE — add/remove models as needed
# ============================================================

MODELS = [
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143", "BNnone_BU_ TD_Skip_Ecoset"),
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731", "BNall32_BU_TD_Skip_Ecoset"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\recurrence_gain_comparison_Ecoset.png"

# ============================================================
# LOAD & COMPUTE
# ============================================================

model_results = []

for log_dir, label in MODELS:
    npz_path = find_npz(log_dir)
    if npz_path is None:
        continue

    if not os.path.exists(npz_path):
        print(f"WARNING: File not found, skipping: {npz_path}")
        continue

    data = np.load(npz_path)

    if "val_accuracies_all" not in data:
        print(f"WARNING: 'val_accuracies_all' not found in {npz_path}, skipping")
        continue

    val = data["val_accuracies_all"]  # shape: (epochs, timesteps)

    t1 = val[:, 0]
    tmax = val.max(axis=1)

    delta_abs = tmax - t1
    delta_pct = (delta_abs / tmax) * 100

    model_results.append({
        "label": label,
        "avg_abs": delta_abs.mean(),
        "max_abs": delta_abs.max(),
        "avg_pct": delta_pct.mean(),
        "max_pct": delta_pct.max(),
    })

    print(f"{label}: Avg Δ={delta_abs.mean():.2f}pp, Max Δ={delta_abs.max():.2f}pp, "
          f"Avg Δ%={delta_pct.mean():.1f}%, Max Δ%={delta_pct.max():.1f}%")

if len(model_results) == 0:
    print("ERROR: No valid models loaded.")
    exit(1)

# ============================================================
# PLOT — grouped bar chart
# ============================================================

labels = [r["label"] for r in model_results]
avg_abs = [r["avg_abs"] for r in model_results]
max_abs = [r["max_abs"] for r in model_results]
avg_pct = [r["avg_pct"] for r in model_results]
max_pct = [r["max_pct"] for r in model_results]

metric_labels = ["Avg Δ (pp)", "Max Δ (pp)", "Avg Δ (%)", "Max Δ (%)"]
n_metrics = len(metric_labels)
n_models = len(labels)

COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#9b5de5", "#f15bb5"]

bar_width = 0.55 / n_models
x = np.arange(n_metrics)

fig, ax = plt.subplots(figsize=(3 + 2 * n_metrics, 5))

for i, (label, a_abs, m_abs, a_pct, m_pct) in enumerate(
        zip(labels, avg_abs, max_abs, avg_pct, max_pct)):
    values = [a_abs, m_abs, a_pct, m_pct]
    offset = (i - (n_models - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values, bar_width, label=label,
                  color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylabel("Value")
ax.set_title("Recurrence Gain Comparison", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
