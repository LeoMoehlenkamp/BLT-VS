"""
Generate a summary table of all trained models with their family and validation accuracy.

Usage:
    python generate_model_table.py --root "C:/Users/moehl/Logs/Final"

Recursively finds all loss_*.npz files under the root directory, reads the
corresponding args.json to determine the model family (BU / BU-TD / BU-Skip /
BU-TD-Skip), and reports best & final validation accuracy.

Output: printed table + CSV file.
"""

import argparse
import json
import os
import glob
import numpy as np
import csv


def detect_family_from_args(args_path):
    """Determine model family from args.json flags."""
    try:
        with open(args_path, "r") as f:
            args = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    td = args.get("topdown_connections", 0)
    skip = args.get("skip_connections", 0)

    if td and skip:
        return "BU-TD-Skip"
    elif td:
        return "BU-TD"
    elif skip:
        return "BU-Skip"
    else:
        return "BU"


def detect_family_from_path(rel_path):
    """Fallback: infer family from the directory path."""
    parts = rel_path.replace("\\", "/").upper()
    if "BU-TD-SKIP" in parts or "BUTDSKIP" in parts or "BU_TD_SKIP" in parts:
        return "BU-TD-Skip"
    if "BU-TD" in parts or "BUTD" in parts or "BU_TD" in parts:
        return "BU-TD"
    if "BU-SKIP" in parts or "BUSKIP" in parts or "BU_SKIP" in parts:
        return "BU-Skip"
    return "BU"


def short_model_name(run_folder_name):
    """Create a readable short name from the run folder name."""
    name = run_folder_name
    # Strip common prefix
    for prefix in ["blt_vs_bottleneck__miniecoset__ts12__", "resnet50__miniecoset__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip timestamp suffix (e.g. __20260402_123451)
    parts = name.rsplit("__", 1)
    if len(parts) == 2 and len(parts[1]) >= 8 and parts[1][:8].isdigit():
        name = parts[0]
    return name


def main():
    parser = argparse.ArgumentParser(description="Generate model summary table")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing model log folders")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <root>/model_summary.csv)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output_csv = args.output or os.path.join(root, "model_summary.csv")

    # Find all loss_*.npz files
    npz_files = glob.glob(os.path.join(root, "**", "loss_*.npz"), recursive=True)
    npz_files.sort()

    if not npz_files:
        print(f"No loss_*.npz files found under {root}")
        return

    rows = []

    for npz_path in npz_files:
        run_dir = os.path.dirname(npz_path)
        run_folder = os.path.basename(run_dir)
        rel_path = os.path.relpath(run_dir, root)

        # --- Load NPZ ---
        try:
            data = np.load(npz_path)
        except Exception as e:
            print(f"  [WARN] Could not load {npz_path}: {e}")
            continue

        if "val_accuracies" not in data:
            print(f"  [WARN] No val_accuracies in {npz_path}")
            continue

        val_acc = data["val_accuracies"]
        best_val_acc = float(np.max(val_acc))
        best_epoch = int(np.argmax(val_acc)) + 1
        final_val_acc = float(val_acc[-1])
        n_epochs = len(val_acc)

        # --- Detect family ---
        args_path = os.path.join(run_dir, "args.json")
        family = detect_family_from_args(args_path)
        if family is None:
            family = detect_family_from_path(rel_path)

        # --- Model name ---
        model_name = short_model_name(run_folder)

        rows.append({
            "model": model_name,
            "family": family,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "final_val_acc": final_val_acc,
            "n_epochs": n_epochs,
            "rel_path": rel_path,
        })

    # Sort by family then best_val_acc descending
    rows.sort(key=lambda r: (r["family"], -r["best_val_acc"]))

    # --- Print table ---
    header = f"{'Model':<45} {'Family':<14} {'Best Val Acc':>12} {'Best Ep':>8} {'Final Val Acc':>14} {'Epochs':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    current_family = None
    for r in rows:
        if r["family"] != current_family:
            if current_family is not None:
                print(sep)
            current_family = r["family"]
        print(f"{r['model']:<45} {r['family']:<14} {r['best_val_acc']:>11.2f}% {r['best_epoch']:>8} {r['final_val_acc']:>13.2f}% {r['n_epochs']:>7}")

    print(sep)
    print(f"\nTotal models: {len(rows)}")

    # --- Write CSV ---
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "family", "best_val_acc",
                                                "best_epoch", "final_val_acc",
                                                "n_epochs", "rel_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
