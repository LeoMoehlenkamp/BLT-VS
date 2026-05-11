"""
Rectangular second-order RDM analysis: Monkey neural RDMs vs. ResNet layer RDMs.

Two modes:
  --resnet_rdm_pkl  : Load ResNet RDMs from TIMM .pkl (PCA-reduced, matches notebook)
  --resnet_rdm_npz  : Load ResNet RDMs from your own extraction .npz

If both are given, --resnet_rdm_pkl takes priority.

Usage:
  # Mode 1: match notebook exactly (TIMM pkl)
  python second_order_rdms_resnet_vs_monkey.py \
      --resnet_rdm_pkl <path_to_resnet_rdms.pkl>

  # Mode 2: test your own extraction (npz)
  python second_order_rdms_resnet_vs_monkey.py \
      --resnet_rdm_npz <path_to_resnet_rdms.npz> --metric cosine --rdm_type ranked
"""

import os
from os import path
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, cdist
from scipy.stats import rankdata


# Default paths
DEFAULT_MONKEY_PKL = (
    "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_mua_minithings/"
    "monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3"
    "-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16"
    "-baseline_0-standardize_1-metric_correlation-neural_mua.pkl"
)
DEFAULT_RESNET_PKL = (
    "/share/klab/danthes/danthes/THINGS_Drift/datasets/TIMM/resnet18/"
    "rdms-resnet18-metric_cosine-normalization_None-pca_1000.pkl"
)
DEFAULT_STIMULUS_CSV = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"

# Colors per ResNet stage
STAGE_COLORS = {
    "conv1":   "#1f77b4",
    "layer1":  "#ff7f0e",
    "layer2":  "#2ca02c",
    "layer3":  "#d62728",
    "layer4":  "#9467bd",
    "avgpool": "#8c564b",
}


def get_layer_color(layer_name):
    for stage in ["layer4", "layer3", "layer2", "layer1", "avgpool", "conv1"]:
        if layer_name.startswith(stage):
            return STAGE_COLORS[stage]
    return "#333333"


def get_stage(layer_name):
    for stage in ["layer4", "layer3", "layer2", "layer1", "avgpool", "conv1"]:
        if layer_name.startswith(stage):
            return stage
    return layer_name


# ============================================================
# Helpers (same as in second_order_rdms_ann_vs_monkey.py)
# ============================================================

def get_rdm_design_sort_indices(stimulus_csv, reduce_to_column="category"):
    stim_info = pd.read_csv(stimulus_csv)
    stim_info_sorted = stim_info.sort_values(
        [
            "animate", "body_parts", "human", "mammal", "non_mammal",
            "inanimate", "natural", "food", "fruit", "vegetable",
            "other_food", "plants", "other_natural", "artificial",
            "artificial_small", "tools", "artificial_small_other",
            "artificial_large", "furniture", "vehicles", "outside_large",
            "cat_id",
        ],
        ascending=False,
    )
    stim_info_select = stim_info_sorted[reduce_to_column]
    stim_info_select = stim_info_select.drop_duplicates()
    sort_idx = rankdata(stim_info_select.index.values).astype(int) - 1
    return sort_idx


def correlate_rdm_movie_with_models(rdm_timecourse, target_rdms, model_keys):
    print(f"  monkey timecourse shape: {rdm_timecourse.shape}")
    models = np.array([target_rdms[key] for key in model_keys], dtype=np.float64)
    print(f"  model rdms shape:        {models.shape}")
    return 1 - cdist(rdm_timecourse, models, metric="correlation")


def plot_rectangular_matrix(
    matrix, row_labels, col_labels, save_path, title,
    xlabel, ylabel, vmin=-1, vmax=1, cmap="Reds",
    figsize=None, x_group_boundaries=None,
):
    n_rows, n_cols = matrix.shape
    if figsize is None:
        fig_w = max(12, n_cols * 0.35)
        fig_h = max(4, n_rows * 0.4)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=max(4, min(7, 200 // max(n_cols, 1))))
    ax.set_xlabel(xlabel)

    if x_group_boundaries is not None:
        for b in x_group_boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rectangular second-order RDM: Monkey vs pretrained ResNet"
    )
    parser.add_argument("--resnet_rdm_pkl", type=str, default=None,
                        help="Path to ResNet RDMs .pkl (TIMM format, matches notebook)")
    parser.add_argument("--resnet_rdm_npz", type=str, default=None,
                        help="Path to ResNet RDMs .npz (from your own extraction)")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric (only used with --resnet_rdm_npz)")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"],
                        help="Raw or ranked RDMs (only used with --resnet_rdm_npz)")
    parser.add_argument("--monkey_pkl_path", type=str, default=DEFAULT_MONKEY_PKL,
                        help="Path to monkey RDM .pkl")
    parser.add_argument("--stimulus_csv", type=str, default=DEFAULT_STIMULUS_CSV,
                        help="Path to stimulus_information.csv")
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/second_order_resnet_vs_monkey")
    parser.add_argument("--plot_panels", type=int, default=1)
    args = parser.parse_args()

    if args.resnet_rdm_pkl is None and args.resnet_rdm_npz is None:
        args.resnet_rdm_pkl = DEFAULT_RESNET_PKL
        print(f"No ResNet RDM path given, using default pkl: {DEFAULT_RESNET_PKL}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load monkey RDMs — use ALL timepoints, NO rank transform
    # (matches notebook exactly)
    # ---------------------------------------------------------
    print(f"Loading monkey RDMs: {args.monkey_pkl_path}")
    with open(args.monkey_pkl_path, "rb") as f:
        monkey_rdm_data = pickle.load(f)

    monkey_time = np.array(monkey_rdm_data["time"])
    monkey_rdms_raw = monkey_rdm_data["rdms"]
    print(f"Monkey RDMs shape: {monkey_rdms_raw.shape}")
    print(f"Monkey timepoints: {len(monkey_time)} ({monkey_time[0]} to {monkey_time[-1]} ms)")

    monkey_labels = [f"M {int(t)}ms" for t in monkey_time]

    # ---------------------------------------------------------
    # Load ResNet RDMs — two modes: pkl (notebook) or npz (own extraction)
    # ---------------------------------------------------------
    model_rdm_dict = {}
    all_resnet_keys = []

    if args.resnet_rdm_pkl is not None:
        # --- Mode 1: TIMM .pkl (matches notebook exactly) ---
        print(f"\nLoading ResNet RDMs from pkl: {args.resnet_rdm_pkl}")
        with open(args.resnet_rdm_pkl, "rb") as f:
            model_rdm_data = pickle.load(f)

        all_resnet_keys = model_rdm_data["selected_nodes"]
        model_rdm_dict = model_rdm_data["rdms"]
        resnet_variant = path.basename(args.resnet_rdm_pkl).split("-")[1] if "-" in path.basename(args.resnet_rdm_pkl) else "resnet"
        source_tag = "pkl"

    else:
        # --- Mode 2: own .npz extraction ---
        print(f"\nLoading ResNet RDMs from npz: {args.resnet_rdm_npz}")
        resnet_data = np.load(args.resnet_rdm_npz, allow_pickle=True)
        resnet_variant = str(resnet_data["resnet_variant"]) if "resnet_variant" in resnet_data else "resnet"
        available_layers = list(resnet_data["layers"]) if "layers" in resnet_data else []

        for layer in available_layers:
            key = f"{layer}_rdm_{args.metric}_{args.rdm_type}"
            if key not in resnet_data:
                print(f"  Warning: {key} not found, skipping")
                continue
            rdm = resnet_data[key].astype(np.float64)
            if rdm.ndim == 2:
                rdm = squareform(rdm)
            model_rdm_dict[layer] = rdm
            all_resnet_keys.append(layer)

        source_tag = f"npz_{args.metric}_{args.rdm_type}"

    if not all_resnet_keys:
        raise ValueError("No ResNet RDMs found.")

    print(f"ResNet variant: {resnet_variant}")
    print(f"ResNet layers loaded: {len(all_resnet_keys)}")
    print(f"  Layers: {all_resnet_keys}")

    # Group layers by stage for block boundaries
    stage_groups = {}
    for layer in all_resnet_keys:
        stage = get_stage(layer)
        stage_groups.setdefault(stage, []).append(layer)

    # Build x-boundaries between stages
    x_boundaries = []
    offset = 0
    stages_seen = []
    for layer in all_resnet_keys:
        stage = get_stage(layer)
        if stage not in stages_seen:
            if stages_seen:
                x_boundaries.append(offset)
            stages_seen.append(stage)
        offset += 1

    # ---------------------------------------------------------
    # Prepare output dirs
    # ---------------------------------------------------------
    run_tag = f"{resnet_variant}_vs_monkey__{source_tag}"
    out_dir = path.join(args.save_dir, run_tag)
    npz_dir = path.join(out_dir, "npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Full correlation: Monkey × ResNet layers
    # (matches notebook: 1 - cdist(rdms, model_rdm_dict, "correlation"))
    # ---------------------------------------------------------
    print("\n=== Correlating monkey timecourse with all ResNet layers ===")
    full_corr = correlate_rdm_movie_with_models(
        monkey_rdms_raw,
        model_rdm_dict,
        all_resnet_keys
    )

    np.savez_compressed(
        path.join(npz_dir, "rectangular_monkey_vs_resnet_similarity.npz"),
        similarity_matrix=full_corr.astype(np.float32),
        row_labels=np.array(monkey_labels),
        col_labels=np.array(all_resnet_keys),
        monkey_times=monkey_time,
    )
    print("  Saved npz/rectangular_monkey_vs_resnet_similarity.npz")

    # ---------------------------------------------------------
    # Stop if no plots
    # ---------------------------------------------------------
    if not args.plot_panels:
        print("\nDone. Plots skipped.")
        return

    print("\n=== Generating plots ===")

    # ---------------------------------------------------------
    # Plot 1: Full heatmap — Monkey × ResNet layers
    # ---------------------------------------------------------
    vmin_full = np.min(full_corr)
    vmax_full = np.max(full_corr)

    plot_rectangular_matrix(
        matrix=full_corr,
        row_labels=monkey_labels,
        col_labels=all_resnet_keys,
        save_path=path.join(out_dir, "full_monkey_vs_resnet_heatmap.png"),
        title=f"Second-Order Similarity: Monkey vs {resnet_variant}",
        xlabel="ResNet layer",
        ylabel="Monkey time (ms)",
        vmin=vmin_full, vmax=vmax_full,
        cmap="Reds",
        x_group_boundaries=x_boundaries,
    )
    print("  [SAVED] full_monkey_vs_resnet_heatmap.png")

    # ---------------------------------------------------------
    # Plot 2: Heatmap + best layer dot per stage
    # ---------------------------------------------------------
    n_rows, n_cols = full_corr.shape
    fig_w = max(12, n_cols * 0.35)
    fig_h = max(4, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(full_corr, aspect="auto", interpolation="nearest",
                   cmap="Reds", vmin=vmin_full, vmax=vmax_full)

    # Best layer per stage for each monkey timepoint
    block_offset = 0
    for stage in stages_seen:
        block_size = len(stage_groups[stage])
        for row_idx in range(n_rows):
            block_corrs = full_corr[row_idx, block_offset:block_offset + block_size]
            best_local = np.argmax(block_corrs)
            best_col = block_offset + best_local
            ax.scatter(best_col, row_idx, color="white", edgecolors="black",
                       linewidths=0.5, s=30, zorder=5)
        block_offset += block_size

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(monkey_labels, fontsize=7)
    ax.set_ylabel("Monkey time (ms)")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(all_resnet_keys, rotation=90,
                       fontsize=max(4, min(7, 200 // max(n_cols, 1))))
    ax.set_xlabel("ResNet layer")

    for b in x_boundaries:
        ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(
        f"Monkey vs {resnet_variant} + Best Layer per Stage",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path.join(out_dir, "full_monkey_vs_resnet_best_layer.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  [SAVED] full_monkey_vs_resnet_best_layer.png")

    # ---------------------------------------------------------
    # Plot 3: Line plot — one line per ResNet stage (best sublayer)
    # x-axis = monkey time
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    block_offset = 0
    for stage in stages_seen:
        block_size = len(stage_groups[stage])
        stage_corr = full_corr[:, block_offset:block_offset + block_size]
        best_corr = np.max(stage_corr, axis=1)

        color = STAGE_COLORS.get(stage, None)
        ax.plot(monkey_time, best_corr, label=stage,
                linewidth=2, color=color)
        block_offset += block_size

    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Best correlation with ResNet")
    ax.set_title(f"Monkey vs {resnet_variant} – best match per stage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, "summary_best_corr_per_stage.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  [SAVED] summary_best_corr_per_stage.png")

    # ---------------------------------------------------------
    # Plot 4: Line plot — one line per ResNet sublayer
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for j, layer in enumerate(all_resnet_keys):
        color = get_layer_color(layer)
        ax.plot(monkey_time, full_corr[:, j],
                label=layer, alpha=0.6, linewidth=1, color=color)

    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Monkey vs {resnet_variant} – all sublayers")
    ax.legend(fontsize=4, ncol=max(1, len(all_resnet_keys) // 10), loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, "all_layers_correlation_curves.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  [SAVED] all_layers_correlation_curves.png")

    # ---------------------------------------------------------
    # Plot 5: Overall best correlation per monkey timepoint
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    best_overall = np.max(full_corr, axis=1)

    ax.plot(monkey_time, best_overall, linewidth=2.5)
    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Best correlation with any ResNet layer")
    ax.set_title(f"Best overall Monkey vs {resnet_variant} correlation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.join(out_dir, "summary_best_overall_corr.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  [SAVED] summary_best_overall_corr.png")

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
