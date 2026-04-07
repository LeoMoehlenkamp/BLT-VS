
"""
Resume Training Script for BLT-VS

Loads a previous training run (by run name) and continues training from
either the BEST or LAST checkpoint. All metrics (losses, accuracies) are
appended to the existing history so that final plots cover ALL epochs
from the very beginning.

Usage:
------
    python resume_training.py \
        --run_name "blt_vs__ecoset__ts12__bn-none__20250401_120000" \
        --checkpoint best \
        --n_epochs 20 \
        --learning_rate 5e-4 \
        --batch_size 4

The run_name must match a folder under logs/perf_logs/ and logs/net_params/.
"""

# ============================
# IMPORTS
# ============================

import argparse
import sys
from tqdm import tqdm
import matplotlib
from datetime import datetime
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
import json
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import numpy as np
import time

from helpers.helper_funcs import get_Dataset_loaders, LinearFitScheduler
from models.helper_funcs import get_network_model, get_optimizer, eval_network, compute_accuracy, adaptive_gradient_clipping

# ============================
# ARGUMENTS
# ============================

parser = argparse.ArgumentParser(description='Resume training from a previous run')

parser.add_argument('--run_name', type=str, required=True,
                    help='Name of the previous run folder (under logs/perf_logs/ and logs/net_params/)')
parser.add_argument('--checkpoint', type=str, default='best', choices=['best', 'last'],
                    help='Which checkpoint to resume from: best or last')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of ADDITIONAL epochs to train')
parser.add_argument('--learning_rate', type=float, default=None,
                    help='Learning rate for resumed training (default: use original)')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size (default: use original)')
parser.add_argument('--batch_size_val_test', type=int, default=None,
                    help='Validation/test batch size (default: use original)')
parser.add_argument('--num_workers', type=int, default=None,
                    help='Number of dataloader workers (default: use original)')
parser.add_argument('--grad_clipping', type=int, default=1)
parser.add_argument('--from_epoch', type=int, default=None,
                    help='Truncate history to this epoch before resuming (e.g. 40 to discard later epochs)')

args = parser.parse_args()

# ============================
# RESOLVE PATHS
# ============================

log_path = os.path.join('logs', 'perf_logs', args.run_name)
net_path = os.path.join('logs', 'net_params', args.run_name)

if not os.path.exists(log_path):
    raise FileNotFoundError(f"Log folder not found: {log_path}")
if not os.path.exists(net_path):
    raise FileNotFoundError(f"Net folder not found: {net_path}")

# ============================
# LOAD ORIGINAL CONFIG
# ============================

config_path = os.path.join(log_path, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config not found: {config_path}")

with open(config_path, "r") as f:
    hyp = json.load(f)

print(f"\nResuming run: {args.run_name}")
print(f"Checkpoint: {args.checkpoint.upper()}")
print(f"Original config loaded from: {config_path}")

# Convert augment sets back from lists (JSON stores sets as lists)
if isinstance(hyp['dataset'].get('augment'), list):
    hyp['dataset']['augment'] = set(hyp['dataset']['augment'])
if isinstance(hyp['dataset'].get('augment_val_test'), list):
    hyp['dataset']['augment_val_test'] = set(hyp['dataset']['augment_val_test'])

# Override with user-specified values
if args.batch_size is not None:
    hyp['optimizer']['batch_size'] = args.batch_size
if args.batch_size_val_test is not None:
    hyp['misc']['batch_size_val_test'] = args.batch_size_val_test
if args.num_workers is not None:
    hyp['optimizer']['dataloader']['num_workers_train'] = args.num_workers

hyp['optimizer']['n_epochs'] = args.n_epochs

# Determine resume LR:
#   If user passed --learning_rate, use that exact value.
#   Otherwise default to base_lr / 10 (the original training likely decayed
#   the LR significantly, so starting at full base_lr would be too high).
original_base_lr = hyp['optimizer']['lr']['base_lr']
if args.learning_rate is not None:
    resume_lr = args.learning_rate
else:
    resume_lr = original_base_lr / 10.0
print(f"Resume LR: {resume_lr}  (original base_lr was {original_base_lr})")

# Ensure dataset_mode exists
if 'dataset_mode' not in hyp:
    hyp['dataset_mode'] = 0

net_name = args.run_name

# ============================
# LOAD EXISTING METRICS
# ============================

# Find loss file
loss_files = [f for f in os.listdir(log_path) if f.startswith('loss_') and f.endswith('.npz')]

if len(loss_files) == 0:
    raise FileNotFoundError(f"No loss file found in {log_path}")

loss_file_path = os.path.join(log_path, loss_files[0])
print(f"Loading existing metrics from: {loss_file_path}")

log_data = np.load(loss_file_path, allow_pickle=True)

train_losses = list(log_data['train_loss'])
train_accuracies = list(log_data['train_accuracies'])
val_losses = list(log_data['val_loss'])
val_accuracies = list(log_data['val_accuracies'])

if "val_accuracies_all" in log_data.files:
    val_accuracies_all = list(log_data["val_accuracies_all"])
else:
    val_accuracies_all = []

total_saved_epochs = len(train_losses)
print(f"Loaded {total_saved_epochs} epochs of training history.")

# -------------------------------------------------------
# Auto-determine from_epoch based on checkpoint type
# (unless the user explicitly passed --from_epoch)
# -------------------------------------------------------
if args.from_epoch is not None:
    # User override — use exactly what they specified
    from_epoch = args.from_epoch
    print(f"Using user-specified --from_epoch={from_epoch}")
elif args.checkpoint == 'best' and len(val_accuracies) > 0:
    # BEST checkpoint → truncate to the epoch where best val acc was achieved
    from_epoch = int(np.argmax(val_accuracies)) + 1
    print(f"Auto-detected: BEST checkpoint was epoch {from_epoch} "
          f"(val acc {val_accuracies[from_epoch - 1]:.2f}%)")
else:
    # LAST checkpoint → keep all history
    from_epoch = total_saved_epochs
    print(f"Using LAST checkpoint — keeping all {from_epoch} epochs of history.")

# Truncate history to from_epoch
if from_epoch < total_saved_epochs:
    print(f"Truncating history to first {from_epoch} epochs "
          f"(discarding epochs {from_epoch + 1}-{total_saved_epochs}).")
    train_losses = train_losses[:from_epoch]
    train_accuracies = train_accuracies[:from_epoch]
    val_losses = val_losses[:from_epoch]
    val_accuracies = val_accuracies[:from_epoch]
    if len(val_accuracies_all) > from_epoch:
        val_accuracies_all = val_accuracies_all[:from_epoch]
elif from_epoch > total_saved_epochs:
    print(f"WARNING: from_epoch={from_epoch} but only {total_saved_epochs} epochs in history. "
          f"Using all {total_saved_epochs}.")

previous_epochs = len(train_losses)
print(f"Resuming from epoch {previous_epochs + 1}.")

# Best val acc from history
if len(val_accuracies) > 0:
    best_val_acc = float(np.max(val_accuracies))
    best_epoch = int(np.argmax(val_accuracies)) + 1
    print(f"Previous best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
else:
    best_val_acc = -float("inf")
    best_epoch = -1

# ============================
# HELPER
# ============================

def save_filtered_state_dict(state_dict, save_path):
    filtered = {k: v for k, v in state_dict.items()
                if 'total_ops' not in k and 'total_params' not in k}
    torch.save(filtered, save_path)

# ============================
# LOAD MODEL + WEIGHTS
# ============================

print("\nBuilding model from config...")
train_loader, val_loader, _, hyp = get_Dataset_loaders(hyp, ['train', 'val'])

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Number of train batches: {len(train_loader)}")

net, _ = get_network_model(hyp)
net = net.float()

# Find checkpoint file
if args.checkpoint == 'best':
    weight_files = [f for f in os.listdir(net_path) if 'BEST' in f and f.endswith('.pth')]
else:
    weight_files = [f for f in os.listdir(net_path) if 'LAST' in f and f.endswith('.pth')]

if len(weight_files) == 0:
    raise FileNotFoundError(
        f"No {args.checkpoint.upper()} checkpoint found in {net_path}. "
        f"Available files: {os.listdir(net_path)}"
    )

weight_path = os.path.join(net_path, weight_files[0])
print(f"Loading weights: {weight_path}")

state_dict = torch.load(weight_path, map_location="cpu")
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
net.load_state_dict(new_state_dict)
print("Weights loaded successfully.")

net.train()

# DataParallel
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)
net.to(hyp['optimizer']['device'])

# ============================
# OPTIMIZER, CRITERION, SCHEDULER
# ============================

criterion = nn.CrossEntropyLoss(
    weight=hyp['dataset'].get('class_weights', None),
    label_smoothing=0.1
)
if criterion.weight is not None:
    criterion.weight = criterion.weight.to(hyp['optimizer']['device'])

optimizer = get_optimizer(hyp, net)
scaler = torch.amp.GradScaler("cuda", enabled=hyp['misc']['use_amp'])

# Set LR directly — no warmup, continue at a low LR
for param_group in optimizer.param_groups:
    param_group['lr'] = resume_lr

# Only the adaptive scheduler (halves LR when val loss plateaus)
lr_scheduler = LinearFitScheduler(
    optimizer, num_epochs=5, factor=1./2,
    min_percent_change=1.0, mode='min', verbose=True, patience=2
)

# ============================
# TRAINING LOOP
# ============================

print(f"\n{'='*50}")
print(f"Resuming training for {args.n_epochs} additional epochs")
print(f"Epochs will be numbered {previous_epochs + 1} to {previous_epochs + args.n_epochs}")
print(f"Learning rate: {resume_lr}")
print(f"{'='*50}\n")

for epoch_rel in range(1, args.n_epochs + 1):

    epoch_abs = previous_epochs + epoch_rel  # absolute epoch number
    start = time.time()

    torch.cuda.synchronize()

    train_loss_running = 0.0
    train_acc_running = 0.0
    epoch_running_init_flag = 0

    print(f'LR now: {optimizer.param_groups[0]["lr"]}')

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_abs}",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout
    )

    for images, labels in pbar:

        imgs = images.to(hyp['optimizer']['device'])
        lbls = labels.to(hyp['optimizer']['device'])

        if criterion.weight is not None:
            criterion.weight = criterion.weight.to(imgs.device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
            outputs = net(imgs)

            if epoch_rel == 1 and epoch_running_init_flag == 0:
                print(f"Labels shape: {lbls.shape}")
                print(f"len(outputs) = {len(outputs)}")

            loss = criterion(outputs[0], lbls.long())
            if len(outputs) > 1:
                for t in range(len(outputs) - 1):
                    loss = loss + criterion(outputs[t + 1], lbls.long())
            loss = loss / len(outputs)

        scaler.scale(loss).backward()

        if args.grad_clipping:
            scaler.unscale_(optimizer)
            adaptive_gradient_clipping(net, clip_factor=0.1)

        scaler.step(optimizer)
        scaler.update()

        train_loss_running += loss.item()
        train_acc_running += np.mean(compute_accuracy(outputs, lbls))

        current_acc = np.mean(compute_accuracy(outputs, lbls))
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{current_acc:.2f}%"})

        if epoch_running_init_flag == 0:
            epoch_running_init_flag = 1

    pbar.close()

    train_losses.append(train_loss_running / len(train_loader))
    train_accuracies.append(train_acc_running / len(train_loader))

    # Validation
    val_loss_running, val_acc_running = eval_network(val_loader, net, criterion, hyp)
    net.train()

    val_acc_running = val_acc_running / len(val_loader)
    val_acc_ts = np.array(val_acc_running, dtype=float)
    val_accuracies_all.append(val_acc_ts)

    val_losses.append(val_loss_running / len(val_loader) / len(outputs))
    val_accuracies.append(float(np.mean(val_acc_ts)))

    current_val_acc = val_accuracies[-1]

    # Update best
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        best_epoch = epoch_abs
        print(f"New BEST model at epoch {best_epoch} (val acc = {best_val_acc:.2f}%)")

        if torch.cuda.device_count() > 1:
            save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}_BEST.pth')
        else:
            save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}_BEST.pth')

    ts_string = " | ".join([f"t{i+1}:{acc:.2f}%" for i, acc in enumerate(val_acc_ts)])
    print(f"Val acc per timestep → {ts_string}")
    print(f'Epoch time: {time.time() - start:.2f} seconds')
    print(f'Train loss: {train_losses[-1]:.2f}; acc: {train_accuracies[-1]:.2f}%')
    print(f'Val loss: {val_losses[-1]:.2f}; acc: {val_accuracies[-1]:.2f}%')

    # LR scheduling (adaptive only, no warmup)
    lr_scheduler.step(val_losses[-1])

    # Save metrics every epoch (overwrites with full history)
    np.savez(
        loss_file_path,
        train_loss=train_losses,
        val_loss=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        val_accuracies_all=np.array(val_accuracies_all, dtype=float)
    )

# ============================
# SAVE LAST CHECKPOINT
# ============================

print(f"\nSaving LAST checkpoint (epoch {previous_epochs + args.n_epochs})")

if torch.cuda.device_count() > 1:
    save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}_LAST.pth')
else:
    save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}_LAST.pth')

# ============================
# TEST EVALUATION
# ============================

_, _, test_loader, hyp = get_Dataset_loaders(hyp, ['test'])
net.eval()

if test_loader is not None:
    test_loss_running, test_acc_running = eval_network(test_loader, net, criterion, hyp)
    test_acc = test_acc_running / len(test_loader)
    print(f"Test acc: {test_acc}")

    np.savez(
        loss_file_path,
        train_loss=train_losses,
        val_loss=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        val_accuracies_all=np.array(val_accuracies_all, dtype=float),
        test_accuracies=test_acc
    )
else:
    print("No test loader available.")

# ============================
# PLOTS (FULL HISTORY from epoch 1)
# ============================

print("\nGenerating plots over FULL training history...")

from matplotlib.ticker import MaxNLocator
import pandas as pd

total_epochs = len(train_losses)
epochs_arr = np.arange(1, total_epochs + 1)

train_loss = np.array(train_losses)
val_loss_arr = np.array(val_losses)
train_acc = np.array(train_accuracies)
val_acc = np.array(val_accuracies)

best_val_acc_plot = np.max(val_acc)
best_val_epoch_plot = np.argmax(val_acc) + 1
train_acc_at_best = train_acc[best_val_epoch_plot - 1]

best_val_loss_plot = np.min(val_loss_arr)
best_loss_epoch_plot = np.argmin(val_loss_arr) + 1
train_loss_at_best = train_loss[best_loss_epoch_plot - 1]

# Resume boundary for visual annotation
resume_epoch = previous_epochs + 0.5  # between old and new

# ============================
# ACCURACY PLOT
# ============================

plt.figure(figsize=(8, 5))
plt.plot(epochs_arr, train_acc, label="Train Accuracy")
plt.plot(epochs_arr, val_acc, label="Validation Accuracy")
plt.axvline(resume_epoch, color='orange', linestyle=':', linewidth=2, label=f"Resumed (after epoch {previous_epochs})")
plt.scatter(best_val_epoch_plot, best_val_acc_plot, color='red', zorder=5)

gap = train_acc_at_best - best_val_acc_plot
plt.annotate(
    f"Best Val Acc: {best_val_acc_plot:.2f}%\nEpoch {best_val_epoch_plot}\nGap: {gap:.2f}%",
    (best_val_epoch_plot, best_val_acc_plot),
    textcoords="offset points", xytext=(-60, 20)
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve (Full History + Resumed)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(log_path + "/accuracy_plot.png", dpi=300)
plt.close()

# ============================
# LOSS PLOT
# ============================

plt.figure(figsize=(8, 5))
plt.plot(epochs_arr, train_loss, label="Train Loss")
plt.plot(epochs_arr, val_loss_arr, label="Validation Loss")
plt.axvline(resume_epoch, color='orange', linestyle=':', linewidth=2, label=f"Resumed (after epoch {previous_epochs})")
plt.scatter(best_loss_epoch_plot, best_val_loss_plot, color='green', zorder=5)

plt.annotate(
    f"Lowest Val Loss: {best_val_loss_plot:.4f}\nEpoch {best_loss_epoch_plot}",
    (best_loss_epoch_plot, best_val_loss_plot),
    textcoords="offset points", xytext=(-60, -25)
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Full History + Resumed)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(log_path + "/loss_plot.png", dpi=300)
plt.close()

# ============================
# SUMMARY TABLE
# ============================

summary = pd.DataFrame({
    "Metric": [
        "Total Epochs",
        "Resumed after Epoch",
        "Best Val Accuracy (%)",
        "Train Accuracy @ Best Val Epoch (%)",
        "Validation Loss @ Best Epoch",
        "Train Loss @ Best Val Epoch"
    ],
    "Value": [
        total_epochs,
        previous_epochs,
        round(best_val_acc_plot, 3),
        round(train_acc_at_best, 3),
        round(best_val_loss_plot, 4),
        round(train_loss_at_best, 4)
    ]
})

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis('off')
table = ax.table(cellText=summary.values, colLabels=summary.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.tight_layout()
plt.savefig(log_path + "/training_summary_table.png", dpi=300)
plt.close()

# ============================
# TIMESTEP ACCURACY PLOTS
# ============================

if len(val_accuracies_all) > 0:

    # Best epoch timestep curve
    best_epoch_idx = int(np.argmax(val_accuracies))
    ts_acc = np.array(val_accuracies_all[best_epoch_idx], dtype=float)
    ts_range = np.arange(1, len(ts_acc) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(ts_range, ts_acc, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Validation Accuracy over Timesteps (Best Epoch {best_epoch_idx + 1})")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_path + "/val_accuracy_over_timesteps_best_epoch.png", dpi=300)
    plt.close()

    # 5 evenly spaced epochs
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
        ts_range = np.arange(1, len(ts_acc) + 1)
        plt.plot(ts_range, ts_acc, marker="o", label=f"Epoch {ep}")

    plt.xlabel("Timestep")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy over Timesteps (5 checkpoints)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path + "/val_accuracy_over_timesteps_5epochs.png", dpi=300)
    plt.close()

# ============================
# RECURRENCE ANALYSIS
# ============================

if len(val_accuracies_all) > 0:

    val_all = np.array(val_accuracies_all)
    n_epochs_total, n_timesteps = val_all.shape
    last_epoch_data = val_all[-1]

    t1 = last_epoch_data[0]
    tmax = last_epoch_data.max()
    rec_score = tmax - t1
    percent_gain = (rec_score / t1) * 100 if t1 != 0 else 0

    print(f"\nFinal Epoch: {n_epochs_total}")
    print(f"T1 Accuracy: {t1:.2f}")
    print(f"Tmax Accuracy: {tmax:.2f}")
    print(f"Recurrence Score (Δ): {rec_score:.2f}")
    print(f"Relative Gain: {percent_gain:.2f}%")

    # Timestep curve (final epoch)
    plt.figure()
    plt.plot(range(1, n_timesteps + 1), last_epoch_data, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy over Timesteps (Final Epoch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "timestep_curve_last_epoch.png"), dpi=300)
    plt.close()

    # Epoch x Timestep table
    columns = [f"t{i+1}" for i in range(n_timesteps)] + ["t_max", "Δ (tmax-t1)", "Δ (%)"]
    rows = []
    for e in range(n_epochs_total):
        row = val_all[e]
        t1_e = row[0]
        tmax_e = row.max()
        delta_e = tmax_e - t1_e
        pct_e = (delta_e / t1_e) * 100 if t1_e != 0 else 0
        full_row = list(np.round(row, 2))
        full_row += [round(tmax_e, 2), round(delta_e, 2), round(pct_e, 2)]
        rows.append(full_row)

    fig, ax = plt.subplots(figsize=(14, 0.4 * n_epochs_total + 2))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=columns,
        rowLabels=[f"E{e+1}" for e in range(n_epochs_total)],
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.title("Validation Accuracy – All Epochs (Timestep Summary)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "timestep_table.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Recurrence gain heatmap
    gain = val_all - val_all[:, 0:1]
    plt.figure(figsize=(10, 6))
    im = plt.imshow(gain, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Gain relative to t1 (Accuracy %)")
    plt.xlabel("Timestep")
    plt.ylabel("Epoch")
    plt.title("Recurrence Gain over Training (Relative to t1)")
    plt.xticks(range(n_timesteps), [f"t{i+1}" for i in range(n_timesteps)])
    plt.yticks(range(0, n_epochs_total, max(1, n_epochs_total // 10)))
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "recurrence_gain_heatmap.png"), dpi=300)
    plt.close()

print("\nAll plots saved (covering full training history).")
print(f"Total epochs: {total_epochs} (original: {previous_epochs}, resumed: {args.n_epochs})")
print("Done!")
