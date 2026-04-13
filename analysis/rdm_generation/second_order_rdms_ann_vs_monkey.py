"""
Rectangular second-order RDM analysis: Monkey neural RDMs vs. ANN model RDMs.

This version only computes and plots RECTANGULAR cross-similarity matrices:
  - rows    = monkey RDMs / monkey timepoints
  - columns = ANN RDMs / ANN areas+timestep

It does NOT build combined square monkey+ANN matrices.

Usage:
  python second_order_rdms_ann_vs_monkey_rectangular.py \
      --ann_rdm_path <path_to_ann_rdms.npz> \
      --monkey_pkl_path <path_to_monkey.pkl> \
      --stimulus_csv <path_to_stimulus_information.csv>
"""

import os
from os import path
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, cdist
from scipy.stats import rankdata


AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

AREA_COLORS = {
    "Retina": "#1f77b4",
    "LGN":    "#ff7f0e",
    "V1":     "#2ca02c",
    "V2":     "#d62728",
    "V3":     "#9467bd",
    "V4":     "#8c564b",
    "LOC":    "#e377c2",
}

# Default paths (monkeyF, LFP, IT = rois_3, correlation distance)
DEFAULT_MONKEY_PKL = (
    "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_lfp_minithings/"
    "monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3"
    "-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16"
    "-baseline_0-standardize_1-metric_correlation-neural_lfp.pkl"
)
DEFAULT_STIMULUS_CSV = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"


# ============================================================
# Helpers
# ============================================================

def get_rdm_design_sort_indices(stimulus_csv, reduce_to_column="category"):
    """Get sort indices for RDM design ordering."""
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
    """
    Correlate monkey RDM timecourse with model RDMs.

    rdm_timecourse: (n_monkey, n_pairs)
    target_rdms:    dict label -> condensed RDM vector
    model_keys:     list of ANN labels to extract

    returns: (n_monkey, n_ann)
    """
    print(f"  monkey timecourse shape: {rdm_timecourse.shape}")
    models = np.array([target_rdms[key] for key in model_keys], dtype=np.float64)
    print(f"  ann model rdms shape:   {models.shape}")
    return 1 - cdist(rdm_timecourse, models, metric="correlation")


def extract_matching_keys(npz_file, area, metric, rdm_type):
    """
    Return sorted (timestep, key) pairs for a given area/metric/rdm_type.

    Supports:
      New format: {area}_t{T}_rdm_{metric}_{rdm_type}
      Old format: {area}_t{T}_rdm_ranked_sorted / {area}_t{T}_rdm_sorted
    """
    pattern_new = re.compile(rf"^{area}_t(\d+)_rdm_{metric}_{rdm_type}$")

    old_suffix = "ranked_sorted" if rdm_type == "ranked" else "sorted"
    pattern_old = re.compile(rf"^{area}_t(\d+)_rdm_{old_suffix}$")

    matches = []
    for key in npz_file.files:
        m = pattern_new.match(key) or pattern_old.match(key)
        if m:
            matches.append((int(m.group(1)), key))

    matches.sort(key=lambda x: x[0])
    return matches


def plot_rectangular_matrix(
    matrix,
    row_labels,
    col_labels,
    save_path,
    title,
    xlabel,
    ylabel,
    vmin=-1,
    vmax=1,
    cmap="Reds",
    figsize=None,
    x_group_boundaries=None,
):
    """Plot a rectangular matrix: rows=monkey, cols=ANN."""
    n_rows, n_cols = matrix.shape

    if figsize is None:
        fig_w = max(12, n_cols * 0.35)
        fig_h = max(4, n_rows * 0.4)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_xlabel(xlabel)

    if x_group_boundaries is not None:
        for b in x_group_boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_x_boundaries(ann_keys_by_area, ordered_areas):
    """Build vertical separator positions between ANN areas."""
    boundaries = []
    offset = 0
    for area in ordered_areas:
        if area in ann_keys_by_area:
            offset += len(ann_keys_by_area[area])
            boundaries.append(offset)
    if len(boundaries) > 0:
        boundaries = boundaries[:-1]
    return boundaries


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rectangular second-order RDM analysis: Monkey vs ANN"
    )
    parser.add_argument(
        "--ann_rdm_path",
        type=str,
        required=True,
        help="Path to ANN RDMs (.npz from save_ann_rdms_extended.py)"
    )
    parser.add_argument(
        "--monkey_pkl_path",
        type=str,
        default=DEFAULT_MONKEY_PKL,
        help="Path to monkey RDM .pkl"
    )
    parser.add_argument(
        "--stimulus_csv",
        type=str,
        default=DEFAULT_STIMULUS_CSV,
        help="Path to stimulus_information.csv"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="analysis_outputs/second_order_ann_vs_monkey_rectangular"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric used in ANN first-order RDMs (e.g. cosine)"
    )
    parser.add_argument(
        "--rdm_type",
        type=str,
        default="ranked",
        choices=["raw", "ranked"],
        help="Use raw or ranked first-order RDMs"
    )
    parser.add_argument(
        "--t_start",
        type=int,
        default=0,
        help="Monkey time start (ms)"
    )
    parser.add_argument(
        "--t_end",
        type=int,
        default=400,
        help="Monkey time end (ms, exclusive)"
    )
    parser.add_argument(
        "--t_step",
        type=int,
        default=10,
        help="Monkey time step (ms)"
    )
    parser.add_argument(
        "--plot_panels",
        type=int,
        default=1
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    t_select = np.arange(args.t_start, args.t_end, args.t_step)

    # ---------------------------------------------------------
    # Load monkey RDMs
    # ---------------------------------------------------------
    print(f"Loading monkey RDMs: {args.monkey_pkl_path}")
    with open(args.monkey_pkl_path, "rb") as f:
        monkey_rdm_data = pickle.load(f)

    monkey_time = np.array(monkey_rdm_data["time"])
    print("monkey_time:", monkey_time[:100])
    print("n timepoints:", len(monkey_time))
    print("unique step sizes:", np.unique(np.diff(monkey_time)))
    print("last 20 monkey times:", monkey_time[-20:])
    monkey_rdms_raw = monkey_rdm_data["rdms"]  # list/array of condensed vectors

    sort_idx = get_rdm_design_sort_indices(
        args.stimulus_csv,
        reduce_to_column=monkey_rdm_data["data_cfg"]["labels"]
    )

    monkey_timecourse = []
    monkey_times_used = []

    for t in t_select:
        matches = np.where(monkey_time == t)[0]
        if len(matches) == 0:
            print(f"  Warning: monkey time {t} not found, skipping")
            continue

        idx = matches[0]

        # condensed -> rank -> square -> reorder -> condensed
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
    # Load ANN RDMs
    # ---------------------------------------------------------
    print(f"\nLoading ANN RDMs: {args.ann_rdm_path}")
    ann_data = np.load(args.ann_rdm_path, allow_pickle=True)

    model_name = path.basename(args.ann_rdm_path).replace("_ann_rdms.npz", "").replace(".npz", "")

    ann_rdm_dict = {}
    ann_keys_by_area = {}
    all_ann_keys = []

    for area in AREAS:
        matches = extract_matching_keys(ann_data, area, args.metric, args.rdm_type)
        if len(matches) == 0:
            print(f"  {area}: no matches found")
            continue

        area_keys = []
        for t, npz_key in matches:
            arr = ann_data[npz_key].astype(np.float64)

            # If stored as square matrix -> convert to condensed.
            # If already condensed -> keep as is.
            if arr.ndim == 2:
                rdm_condensed = squareform(arr)
            elif arr.ndim == 1:
                rdm_condensed = arr
            else:
                raise ValueError(f"Unsupported ANN RDM shape for key {npz_key}: {arr.shape}")

            label = f"{area} t{t}"
            ann_rdm_dict[label] = rdm_condensed
            area_keys.append(label)

        ann_keys_by_area[area] = area_keys
        all_ann_keys.extend(area_keys)
        print(f"  {area}: {len(area_keys)} timesteps")

    if len(all_ann_keys) == 0:
        raise ValueError("No ANN RDMs found. Check --metric and --rdm_type.")

    print(f"ANN total RDMs: {len(all_ann_keys)}")

    # ---------------------------------------------------------
    # Prepare output dirs
    # ---------------------------------------------------------
    run_tag = f"{model_name}__{args.metric}_{args.rdm_type}"
    out_dir = path.join(args.save_dir, run_tag)
    npz_dir = path.join(out_dir, "npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    # ---------------------------------------------------------
    # A) Full rectangular cross-correlation: Monkey x ALL ANN
    # ---------------------------------------------------------
    print("\n=== Correlating monkey timecourse with all ANN RDMs ===")
    full_corr = correlate_rdm_movie_with_models(
        monkey_timecourse,
        ann_rdm_dict,
        all_ann_keys
    )
    # Shape: (n_monkey_times, n_all_ann_rdms)

    np.savez_compressed(
        path.join(npz_dir, "rectangular_monkey_vs_ann_similarity.npz"),
        similarity_matrix=full_corr.astype(np.float32),
        row_labels=np.array(monkey_labels),
        col_labels=np.array(all_ann_keys),
        monkey_times=monkey_times_used,
        metric=np.array(args.metric),
        rdm_type=np.array(args.rdm_type),
    )
    print("  Saved npz/rectangular_monkey_vs_ann_similarity.npz")

    # ---------------------------------------------------------
    # B) Per-area rectangular cross-correlation
    # ---------------------------------------------------------
    area_corr_data = {}

    for area in AREAS:
        if area not in ann_keys_by_area:
            continue

        area_keys = ann_keys_by_area[area]
        print(f"\n=== Correlating monkey timecourse with {area} ===")

        area_corr = correlate_rdm_movie_with_models(
            monkey_timecourse,
            ann_rdm_dict,
            area_keys
        )
        # Shape: (n_monkey_times, n_area_timesteps)

        area_timesteps = [int(k.split(" t")[1]) for k in area_keys]

        area_corr_data[area] = {
            "corr": area_corr,
            "ann_timesteps": area_timesteps,
            "ann_keys": area_keys,
        }

        np.savez_compressed(
            path.join(npz_dir, f"{area}_rectangular_cross_correlation.npz"),
            similarity_matrix=area_corr.astype(np.float32),
            row_labels=np.array(monkey_labels),
            col_labels=np.array([f"t{t}" for t in area_timesteps]),
            monkey_times=monkey_times_used,
            ann_timesteps=np.array(area_timesteps, dtype=np.int32),
        )
        print(f"  Saved npz/{area}_rectangular_cross_correlation.npz")

    # ---------------------------------------------------------
    # Stop here if plots are disabled
    # ---------------------------------------------------------
    if not args.plot_panels:
        print("\nDone. Plots skipped.")
        return

    print("\n=== Generating rectangular plots ===")

    x_boundaries = build_x_boundaries(ann_keys_by_area, AREAS)

    # ---------------------------------------------------------
    # Plot 2: Per-area rectangular heatmaps
    # ---------------------------------------------------------
    available_areas = [a for a in AREAS if a in area_corr_data]

    for area in available_areas:
        info = area_corr_data[area]
        cross = info["corr"]
        a_ts = info["ann_timesteps"]

        save_path = path.join(out_dir, f"{area}_rectangular_cross_correlation.png")
        plot_rectangular_matrix(
            matrix=cross,
            row_labels=monkey_labels,
            col_labels=[f"t{t}" for t in a_ts],
            save_path=save_path,
            title=f"{area}: Monkey vs ANN timestep correlation – {model_name}",
            xlabel="ANN timestep",
            ylabel="Monkey time (ms)",
            vmin=-1,
            vmax=1,
            cmap="Reds",
            figsize=(max(5, len(a_ts) * 0.65), max(4, len(monkey_labels) * 0.35)),
        )
        print(f"  [SAVED] {save_path}")

    # ---------------------------------------------------------
    # Plot 3: Per-area line plots
    # x-axis = monkey time
    # one line per ANN timestep
    # ---------------------------------------------------------
    for area in available_areas:
        info = area_corr_data[area]
        cross = info["corr"]  # (n_monkey_times, n_area_timesteps)
        a_ts = info["ann_timesteps"]

        fig, ax = plt.subplots(figsize=(8, 4))
        for j, ann_t in enumerate(a_ts):
            ax.plot(
                monkey_times_used,
                cross[:, j],
                label=f"t{ann_t}",
                alpha=0.8,
                linewidth=1.5
            )

        ax.set_xlabel("Monkey time (ms)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"{area} – ANN timesteps vs monkey RDM timecourse – {model_name}")
        ax.legend(fontsize=7, ncol=max(1, len(a_ts) // 4), loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        line_path = path.join(out_dir, f"{area}_correlation_curves.png")
        plt.savefig(line_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {line_path}")

    # ---------------------------------------------------------
    # Plot 4: Summary line plot – best ANN match per area
    # For each monkey timepoint, take best ANN timestep within each area
    # ---------------------------------------------------------
    if len(available_areas) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))

        for area in available_areas:
            info = area_corr_data[area]
            cross = info["corr"]
            best_corr = np.max(cross, axis=1)

            color = AREA_COLORS.get(area, None)
            ax.plot(
                monkey_times_used,
                best_corr,
                label=area,
                linewidth=2,
                color=color
            )

        ax.set_xlabel("Monkey time (ms)")
        ax.set_ylabel("Best correlation with ANN")
        ax.set_title(f"Best ANN match per area – {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        summary_path = path.join(out_dir, "summary_best_corr_per_area.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {summary_path}")

    # ---------------------------------------------------------
    # Plot 5: Summary line plot – best area+timestep overall
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    best_overall = np.max(full_corr, axis=1)

    ax.plot(monkey_times_used, best_overall, linewidth=2.5)
    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Best correlation with any ANN RDM")
    ax.set_title(f"Best overall Monkey vs ANN correlation – {model_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    overall_path = path.join(out_dir, "summary_best_overall_corr.png")
    plt.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overall_path}")

    # ---------------------------------------------------------
    # Adjusted Big Second-Order Similarity: Rectangular
    # ---------------------------------------------------------
    print("\n=== Generating adjusted big second-order similarity matrices ===")

    # Compute the big second-order similarity matrix (rectangular)
    big_second_order_corr = correlate_rdm_movie_with_models(
        monkey_timecourse, ann_rdm_dict, all_ann_keys
    )
    # Shape: (n_monkey_times, n_ann_total)

    # Save the rectangular second-order similarity matrix
    np.savez_compressed(
        path.join(npz_dir, "big_second_order_similarity_rectangular.npz"),
        second_order_similarity=big_second_order_corr.astype(np.float32),
        row_labels=np.array(monkey_labels),
        col_labels=np.array(all_ann_keys),
        monkey_times=monkey_times_used,
        metric=np.array(args.metric),
        rdm_type=np.array(args.rdm_type),
    )
    print("  Saved npz/big_second_order_similarity_rectangular.npz")

    # Plot the rectangular second-order similarity matrix
    big_second_order_plot_path = path.join(out_dir, "big_second_order_similarity_rectangular.png")
    # ---------------------------------------------------------
    # Adjust color scale dynamically based on data
    # ---------------------------------------------------------
    print("\n=== Adjusting color scale dynamically for plots ===")

    # Update the big second-order similarity plot
    big_second_order_vmin = np.min(big_second_order_corr)
    big_second_order_vmax = np.max(big_second_order_corr)
    plot_rectangular_matrix(
        matrix=big_second_order_corr,
        row_labels=monkey_labels,
        col_labels=all_ann_keys,
        save_path=big_second_order_plot_path,
        title=f"Second-Order Similarity: Monkey vs ANN\n{model_name} ({args.metric}, {args.rdm_type})",
        xlabel="ANN layer / timestep",
        ylabel="Monkey time (ms)",
        vmin=big_second_order_vmin,
        vmax=big_second_order_vmax,
        cmap="Reds",
        x_group_boundaries=x_boundaries,
    )
    print(f"  [UPDATED] {big_second_order_plot_path} with dynamic color scale")

    # ---------------------------------------------------------
    # Plot: Big second-order similarity + best timestep dots
    # For each monkey timepoint (row) and each ANN layer (block),
    # plot a dot at the column index of the best-matching ANN timestep.
    # If dynamics are similar, dots should form diagonals per block.
    # ---------------------------------------------------------
    print("\n=== Generating best-timestep overlay plot ===")

    n_rows, n_cols = big_second_order_corr.shape
    fig_w = max(12, n_cols * 0.35)
    fig_h = max(4, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        big_second_order_corr,
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=big_second_order_vmin,
        vmax=big_second_order_vmax,
    )

    # Compute block offsets for each area
    block_offset = 0
    for area in AREAS:
        if area not in ann_keys_by_area:
            continue
        block_size = len(ann_keys_by_area[area])

        for row_idx in range(n_rows):
            block_corrs = big_second_order_corr[row_idx, block_offset:block_offset + block_size]
            best_local = np.argmax(block_corrs)
            best_col = block_offset + best_local
            ax.scatter(best_col, row_idx, color="white", edgecolors="black",
                       linewidths=0.5, s=30, zorder=5)

        block_offset += block_size

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(monkey_labels, fontsize=7)
    ax.set_ylabel("Monkey time (ms)")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(all_ann_keys, rotation=90, fontsize=5)
    ax.set_xlabel("ANN layer / timestep")

    if x_boundaries is not None:
        for b in x_boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(
        f"Second-Order Similarity + Best Timestep per Layer\n"
        f"{model_name} ({args.metric}, {args.rdm_type})",
        fontsize=11,
    )

    plt.tight_layout()
    overlay_path = path.join(out_dir, "big_second_order_similarity_best_timestep.png")
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overlay_path}")

    # Update per-area plots
    for area in available_areas:
        info = area_corr_data[area]
        cross = info["corr"]
        a_ts = info["ann_timesteps"]

        area_vmin = np.min(cross)
        area_vmax = np.max(cross)
        save_path = path.join(out_dir, f"{area}_rectangular_cross_correlation.png")
        plot_rectangular_matrix(
            matrix=cross,
            row_labels=monkey_labels,
            col_labels=[f"t{t}" for t in a_ts],
            save_path=save_path,
            title=f"{area}: Monkey vs ANN timestep correlation – {model_name}",
            xlabel="ANN timestep",
            ylabel="Monkey time (ms)",
            vmin=area_vmin,
            vmax=area_vmax,
            cmap="Reds",
            figsize=(max(5, len(a_ts) * 0.65), max(4, len(monkey_labels) * 0.35)),
        )
        print(f"  [UPDATED] {save_path} with dynamic color scale")

    print(f"\nDone. All rectangular outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()