"""
Generate loss/accuracy plots from the npz file for the interrupted training run:
blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451

Uses the full 122-epoch data from the main Logs directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

run_name = "blt_vs_bottleneck__miniecoset__ts12__bnall16__20260402_123451"
log_path = os.path.join(r"C:\Users\moehl\Logs", run_name)
npz_path = os.path.join(log_path, f"loss_{run_name}.npz")

data = np.load(npz_path)
train_loss = data['train_loss']
val_loss = data['val_loss']
train_acc = data['train_accuracies']
val_acc = data['val_accuracies']
val_acc_all = data['val_accuracies_all']

n_epochs = len(train_loss)
epochs = np.arange(1, n_epochs + 1)

# The Exp/final copy has 62 epochs — mark where original training ended
resume_epoch = 62

print(f"Total logged epochs: {n_epochs}")
print(f"Original training: 1-{resume_epoch}, Resumed: {resume_epoch+1}-{n_epochs}")
print(f"Configured n_epochs per run: 40 (implied ~{n_epochs} total across runs)")

# ============================
# Compute best values
# ============================
best_val_acc = np.max(val_acc)
best_val_epoch = np.argmax(val_acc) + 1
train_acc_at_best = train_acc[best_val_epoch - 1]

best_val_loss = np.min(val_loss)
best_loss_epoch = np.argmin(val_loss) + 1
train_loss_at_best = train_loss[best_loss_epoch - 1]

print(f"\nBest val accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
print(f"Train accuracy at best val epoch: {train_acc_at_best:.2f}%")
print(f"Lowest val loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
print(f"Last epoch train loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")
print(f"Last epoch train acc: {train_acc[-1]:.2f}%, val acc: {val_acc[-1]:.2f}%")

# ============================
# ACCURACY PLOT
# ============================
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.scatter(best_val_epoch, best_val_acc, color='red', zorder=5)
plt.axvline(best_val_epoch, linestyle='--', alpha=0.5, color='red')
plt.axvline(resume_epoch, linestyle=':', alpha=0.4, color='gray', label=f"Resume point (epoch {resume_epoch})")

gap = train_acc_at_best - best_val_acc
plt.annotate(
    f"Best Val Acc: {best_val_acc:.2f}%\nEpoch {best_val_epoch}\nGap: {gap:.2f}%",
    (best_val_epoch, best_val_acc),
    textcoords="offset points",
    xytext=(-80, 20)
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"Accuracy Curve — {run_name}\n({n_epochs} epochs logged, configured for 40/run)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
out_acc = os.path.join(log_path, "accuracy_plot_full.png")
plt.savefig(out_acc, dpi=300)
plt.close()
print(f"\nSaved: {out_acc}")

# ============================
# LOSS PLOT
# ============================
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.scatter(best_loss_epoch, best_val_loss, color='green', zorder=5)
plt.axvline(best_loss_epoch, linestyle='--', alpha=0.5, color='green')
plt.axvline(resume_epoch, linestyle=':', alpha=0.4, color='gray', label=f"Resume point (epoch {resume_epoch})")

plt.annotate(
    f"Lowest Val Loss: {best_val_loss:.4f}\nEpoch {best_loss_epoch}",
    (best_loss_epoch, best_val_loss),
    textcoords="offset points",
    xytext=(-80, -30)
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss Curve — {run_name}\n({n_epochs} epochs logged)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
out_loss = os.path.join(log_path, "loss_plot_full.png")
plt.savefig(out_loss, dpi=300)
plt.close()
print(f"Saved: {out_loss}")

# ============================
# SUMMARY TABLE
# ============================
cell_text = [
    ["Total Epochs Logged", n_epochs],
    ["Best Val Accuracy (%)", round(best_val_acc, 3)],
    ["Best Val Accuracy Epoch", best_val_epoch],
    ["Train Accuracy @ Best Val Epoch (%)", round(train_acc_at_best, 3)],
    ["Lowest Val Loss", round(best_val_loss, 4)],
    ["Lowest Val Loss Epoch", best_loss_epoch],
    ["Train Loss @ Best Val Epoch", round(train_loss_at_best, 4)],
    ["Final Train Loss", round(float(train_loss[-1]), 4)],
    ["Final Val Loss", round(float(val_loss[-1]), 4)],
    ["Final Train Accuracy (%)", round(float(train_acc[-1]), 2)],
    ["Final Val Accuracy (%)", round(float(val_acc[-1]), 2)],
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
table = ax.table(
    cellText=cell_text,
    colLabels=["Metric", "Value"],
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.tight_layout()
out_table = os.path.join(log_path, "training_summary_table_full.png")
plt.savefig(out_table, dpi=300)
plt.close()
print(f"Saved: {out_table}")

# ============================
# TIMESTEP ACCURACY (Best Epoch)
# ============================
if val_acc_all.shape[0] > 0:
    best_idx = int(np.argmax(val_acc))
    ts_acc = val_acc_all[best_idx]
    timesteps = np.arange(1, len(ts_acc) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, ts_acc, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Validation Accuracy over Timesteps (Best Epoch {best_idx + 1})")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_ts = os.path.join(log_path, "val_accuracy_over_timesteps_best_epoch_full.png")
    plt.savefig(out_ts, dpi=300)
    plt.close()
    print(f"Saved: {out_ts}")

print("\nDone!")
