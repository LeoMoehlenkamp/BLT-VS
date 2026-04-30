"""
Rectangular second-order RDM analysis: Monkey neural RDMs vs. ResNet layer RDMs.

Identical logic to second_order_rdms_ann_vs_monkey.py, but using ResNet layers
instead of BLT-VS areas+timesteps:
  - rows    = monkey timepoints
  - columns = ResNet layers (conv1_bn, layer1.0.bn1, ..., avgpool)

Usage:
  python second_order_rdms_resnet_vs_monkey.py \
      --resnet_rdm_path <path_to_resnet_rdms.npz> \
      --monkey_pkl_path <path_to_monkey.pkl> \
      --stimulus_csv <path_to_stimulus_information.csv>
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


# Default paths (monkeyF, LFP, IT = rois_3, correlation distance)
DEFAULT_MONKEY_PKL = (
    "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_lfp_minithings/"
    "monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3"
    "-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16"
    "-baseline_0-standardize_1-metric_correlation-neural_lfp.pkl"
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
    parser.add_argument("--resnet_rdm_path", type=str, required=True,
                        help="Path to ResNet RDMs (.npz)")
    parser.add_argument("--monkey_pkl_path", type=str, default=DEFAULT_MONKEY_PKL,
                        help="Path to monkey RDM .pkl")
    parser.add_argument("--stimulus_csv", type=str, default=DEFAULT_STIMULUS_CSV,
                        help="Path to stimulus_information.csv")
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/second_order_resnet_vs_monkey")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric used in ResNet first-order RDMs")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"],
                        help="Use raw or ranked first-order RDMs")
    parser.add_argument("--t_start", type=int, default=0)
    parser.add_argument("--t_end", type=int, default=400)
    parser.add_argument("--t_step", type=int, default=10)
    parser.add_argument("--plot_panels", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    t_select = np.arange(args.t_start, args.t_end, args.t_step)

    # ---------------------------------------------------------
    # Load monkey RDMs (identical to ann_vs_monkey pipeline)
    # ---------------------------------------------------------
    print(f"Loading monkey RDMs: {args.monkey_pkl_path}")
    with open(args.monkey_pkl_path, "rb") as f:
        monkey_rdm_data = pickle.load(f)

    monkey_time = np.array(monkey_rdm_data["time"])
    print(f"n timepoints: {len(monkey_time)}")
    monkey_rdms_raw = monkey_rdm_data["rdms"]

    sort_idx = get_rdm_design_sort_indices(
        args.stimulus_csv,
        reduce_to_column=monkey_rdm_data["data_cfg"]["labels"]
    )

    monkey_timecourse = []
    monkey_times_used = []

    for t in t_select:
        matches = np.where(monkey_time == t)[0]
        if len(matches) == 0:
            continue
        idx = matches[0]

        rdm = monkey_rdms_raw[idx].astype(np.float64)
        if args.rdm_type == "ranked":
            rdm = rankdata(rdm)

        rdm = squareform(rdm)
        rdm = rdm[sort_idx][:, sort_idx]
        rdm = squareform(rdm)

        monkey_timecourse.append(rdm)
        monkey_times_used.append(t)

    monkey_timecourse = np.array(monkey_timecourse, dtype=np.float64)
    monkey_times_used = np.array(monkey_times_used, dtype=np.int32)
    monkey_labels = [f"M {int(t)}ms" for t in monkey_times_used]

    print(f"Monkey timecourse shape: {monkey_timecourse.shape}")
    print(f"Monkey timepoints used:  {monkey_times_used}")

    if monkey_timecourse.shape[0] == 0:
        raise ValueError("No monkey timepoints available after selection.")

    # ---------------------------------------------------------
    # Load ResNet RDMs
    # ---------------------------------------------------------
    print(f"\nLoading ResNet RDMs: {args.resnet_rdm_path}")
    resnet_data = np.load(args.resnet_rdm_path, allow_pickle=True)

    resnet_variant = str(resnet_data["resnet_variant"]) if "resnet_variant" in resnet_data else "resnet"
    available_layers = list(resnet_data["layers"]) if "layers" in resnet_data else []

    resnet_rdm_dict = {}
    resnet_labels = []

    for layer in available_layers:
        key = f"{layer}_rdm_{args.metric}_{args.rdm_type}"
        if key not in resnet_data:
            print(f"  Warning: {key} not found, skipping")
            continue

        rdm = resnet_data[key].astype(np.float64)
        if rdm.ndim == 2:
            rdm_vec = squareform(rdm)
        else:
            rdm_vec = rdm

        # Apply same sort_idx reordering as monkey for consistency
        rdm_sq = squareform(rdm_vec) if rdm_vec.ndim == 1 else rdm
        rdm_sorted = rdm_sq[sort_idx][:, sort_idx]
        rdm_vec_sorted = squareform(rdm_sorted)

        resnet_rdm_dict[layer] = rdm_vec_sorted
        resnet_labels.append(layer)

    if not resnet_labels:
        raise ValueError("No ResNet RDMs found. Check --metric / --rdm_type.")

    print(f"ResNet layers loaded: {len(resnet_labels)}")
    all_resnet_keys = resnet_labels

    # Group layers by stage for block boundaries
    stage_groups = {}
    for layer in resnet_labels:
        stage = get_stage(layer)
        stage_groups.setdefault(stage, []).append(layer)

    # Build x-boundaries between stages
    x_boundaries = []
    offset = 0
    stages_seen = []
    for layer in resnet_labels:
        stage = get_stage(layer)
        if stage not in stages_seen:
            if stages_seen:
                x_boundaries.append(offset)
            stages_seen.append(stage)
        offset += 1

    # ---------------------------------------------------------
    # Prepare output dirs
    # ---------------------------------------------------------
    run_tag = f"{resnet_variant}_vs_monkey__{args.metric}_{args.rdm_type}"
    out_dir = path.join(args.save_dir, run_tag)
    npz_dir = path.join(out_dir, "npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Full rectangular cross-correlation: Monkey × ResNet layers
    # ---------------------------------------------------------
    print("\n=== Correlating monkey timecourse with all ResNet layers ===")
    full_corr = correlate_rdm_movie_with_models(
        monkey_timecourse,
        resnet_rdm_dict,
        all_resnet_keys
    )

    np.savez_compressed(
        path.join(npz_dir, "rectangular_monkey_vs_resnet_similarity.npz"),
        similarity_matrix=full_corr.astype(np.float32),
        row_labels=np.array(monkey_labels),
        col_labels=np.array(all_resnet_keys),
        monkey_times=monkey_times_used,
        metric=np.array(args.metric),
        rdm_type=np.array(args.rdm_type),
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
        title=f"Second-Order Similarity: Monkey vs {resnet_variant}\n({args.metric}, {args.rdm_type})",
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
        f"Monkey vs {resnet_variant} + Best Layer per Stage\n"
        f"({args.metric}, {args.rdm_type})",
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
        ax.plot(monkey_times_used, best_corr, label=stage,
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

    for j, layer in enumerate(resnet_labels):
        color = get_layer_color(layer)
        ax.plot(monkey_times_used, full_corr[:, j],
                label=layer, alpha=0.6, linewidth=1, color=color)

    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Monkey vs {resnet_variant} – all sublayers")
    ax.legend(fontsize=4, ncol=max(1, len(resnet_labels) // 10), loc="best")
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

    ax.plot(monkey_times_used, best_overall, linewidth=2.5)
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
