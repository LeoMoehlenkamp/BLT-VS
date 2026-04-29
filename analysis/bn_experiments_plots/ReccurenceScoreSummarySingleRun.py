import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ---- CLI ARGS (with local default) ----
parser = argparse.ArgumentParser()
parser.add_argument("--npz_path", type=str,
                    default=r"C:\Users\moehl\Logs\Final\BU-TD\BNnone_BU_TD\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158\loss_blt_vs_bottleneck__miniecoset__ts12__bnall96_BU-TD__20260421_120158.npz")
args = parser.parse_args()

npz_path = args.npz_path

# ---- LOAD DATA ----
data = np.load(npz_path)
val = data["val_accuracies_all"]  # shape: (epochs, timesteps)

# ---- COMPUTE METRICS ----
t1 = val[:, 0]
tmax = val.max(axis=1)

delta_abs = tmax - t1
delta_pct = (delta_abs / tmax) * 100

avg_abs = delta_abs.mean()
max_abs = delta_abs.max()
avg_pct = delta_pct.mean()
max_pct = delta_pct.max()

print("Avg Δ (abs):", avg_abs)
print("Max Δ (abs):", max_abs)
print("Avg Δ (%):", avg_pct)
print("Max Δ (%):", max_pct)

# ---- PLOT ----
labels = ["Avg Δ", "Max Δ", "Avg Δ (%)", "Max Δ (%)"]
values = [avg_abs, max_abs, avg_pct, max_pct]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values)

# Add values on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}",
             ha='center', va='bottom')

plt.title("Recurrence Gain Summary")
plt.ylabel("Value")
plt.tight_layout()

# ---- SAVE IN SAME FOLDER ----
save_dir = os.path.dirname(npz_path)
save_path = os.path.join(save_dir, "recurrence_gain_summary.png")

plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved to: {save_path}")