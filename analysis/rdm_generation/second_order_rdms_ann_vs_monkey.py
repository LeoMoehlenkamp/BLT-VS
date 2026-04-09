"""
Second-order RDM analysis: ANN model layers/timesteps vs. monkey neural RDMs.

Follows the approach from the reference notebook:
  - Load monkey RDM .pkl directly → build ranked sorted timecourse
  - Load ANN RDMs .npz → extract ranked sorted condensed vectors
  - correlate_rdm_movie_with_models via cdist (Pearson on ranked vectors)
  - Produce second-order RDMs + correlation plots

Usage:
  python second_order_rdms_ann_vs_monkey.py \
      --ann_rdm_path  <path_to_ann_rdms.npz> \
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

FULL_PANEL_SIZE = (24, 6)

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
# Functions (matching reference notebook + save_monkey_rdms.py)
# ============================================================

def get_rdm_design_sort_indices(stimulus_csv, reduce_to_column="category"):
    """Get sort indices for RDM design ordering (from save_monkey_rdms.py)."""
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
    Correlate monkey RDM timecourse with model RDMs (from reference notebook).

    rdm_timecourse: (n_times, n_pairs) – ranked sorted condensed monkey RDM vectors
    target_rdms:    dict  key → condensed RDM vector
    model_keys:     list of keys to select from target_rdms

    Returns: (n_times, n_models) Pearson correlation matrix
    """
    print(f"  time course shape: {rdm_timecourse.shape}")
    models = np.array([target_rdms[key] for key in model_keys])
    print(f"  models shape: {models.shape}")
    return 1 - cdist(rdm_timecourse, models, metric="correlation")


def extract_matching_keys(npz_file, area, metric, rdm_type):
    """
    Return sorted (timestep, key) pairs for a given area/metric/rdm_type.

    Supports two key formats:
      New (save_ann_rdms_extended.py): {area}_t{T}_rdm_{metric}_{rdm_type}
      Old (save_ann_rdms.py):          {area}_t{T}_rdm_ranked_sorted  /  {area}_t{T}_rdm_sorted
    """
    # New format: includes metric name
    pattern_new = re.compile(rf"^{area}_t(\d+)_rdm_{metric}_{rdm_type}$")

    # Old format: no metric, uses 'ranked_sorted' or 'sorted'
    old_suffix = "ranked_sorted" if rdm_type == "ranked" else "sorted"
    pattern_old = re.compile(rf"^{area}_t(\d+)_rdm_{old_suffix}$")

    matches = []
    for key in npz_file.files:
        m = pattern_new.match(key) or pattern_old.match(key)
        if m:
            matches.append((int(m.group(1)), key))

    matches.sort(key=lambda x: x[0])
    return matches


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Second-order RDM: ANN layers/timesteps vs monkey neural RDMs"
    )
    parser.add_argument("--ann_rdm_path", type=str, required=True,
                        help="Path to ANN RDMs (.npz from save_ann_rdms_extended.py)")
    parser.add_argument("--monkey_pkl_path", type=str, default=DEFAULT_MONKEY_PKL,
                        help="Path to monkey RDM .pkl (default: monkeyF LFP IT)")
    parser.add_argument("--stimulus_csv", type=str, default=DEFAULT_STIMULUS_CSV,
                        help="Path to stimulus_information.csv")
    parser.add_argument("--save_dir", type=str, default="analysis_outputs/second_order_ann_vs_monkey")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric used in ANN first-order RDMs (e.g. cosine)")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"],
                        help="Use raw or ranked first-order RDMs")
    parser.add_argument("--t_start", type=int, default=0,
                        help="Monkey time start (ms)")
    parser.add_argument("--t_end", type=int, default=160,
                        help="Monkey time end (ms, exclusive)")
    parser.add_argument("--t_step", type=int, default=10,
                        help="Monkey time step (ms)")
    parser.add_argument("--plot_panels", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    t_select = np.arange(args.t_start, args.t_end, args.t_step)

    # ---------------------------------------------------------
    # Load monkey pkl (like reference notebook + save_monkey_rdms.py)
    # ---------------------------------------------------------
    print(f"Loading monkey RDMs: {args.monkey_pkl_path}")
    with open(args.monkey_pkl_path, "rb") as f:
        monkey_rdm_data = pickle.load(f)

    monkey_time = np.array(monkey_rdm_data["time"])
    monkey_rdms_raw = monkey_rdm_data["rdms"]  # list of condensed vectors

    sort_idx = get_rdm_design_sort_indices(
        args.stimulus_csv,
        reduce_to_column=monkey_rdm_data["data_cfg"]["labels"]
    )

    # Build monkey RDM timecourse: ranked, sorted condensed vectors
    # (same processing as in save_monkey_rdms.py panel plot)
    monkey_timecourse = []
    monkey_times_used = []

    for t in t_select:
        matches = np.where(monkey_time == t)[0]
        if len(matches) == 0:
            print(f"  Warning: monkey time {t} not found, skipping")
            continue

        idx = matches[0]
        rdm = monkey_rdms_raw[idx]       # condensed vector
        rdm = rankdata(rdm)              # rank
        rdm = squareform(rdm)            # to square
        rdm = rdm[sort_idx][:, sort_idx] # sort by category design
        rdm = squareform(rdm)            # back to condensed
        monkey_timecourse.append(rdm)
        monkey_times_used.append(t)

    monkey_timecourse = np.array(monkey_timecourse, dtype=np.float64)
    monkey_times_used = np.array(monkey_times_used)
    monkey_labels = [f"M {int(t)}ms" for t in monkey_times_used]

    print(f"Monkey timecourse: {monkey_timecourse.shape}")
    print(f"Monkey timepoints used: {monkey_times_used}")

    # ---------------------------------------------------------
    # Load ANN RDMs → build dict (like model_rdm_dict in notebook)
    # ---------------------------------------------------------
    print(f"\nLoading ANN RDMs: {args.ann_rdm_path}")
    ann_data = np.load(args.ann_rdm_path, allow_pickle=True)

    model_name = path.basename(args.ann_rdm_path).replace("_ann_rdms.npz", "").replace(".npz", "")

    # Build flat dict: key → condensed vector (like model_rdm_dict[key] in notebook)
    ann_rdm_dict = {}
    ann_keys_by_area = {}
    all_ann_keys = []

    for area in AREAS:
        matches = extract_matching_keys(ann_data, area, args.metric, args.rdm_type)
        if len(matches) == 0:
            continue

        area_keys = []
        for t, npz_key in matches:
            rdm_square = ann_data[npz_key].astype(np.float64)  # already ranked + sorted
            rdm_condensed = squareform(rdm_square)
            label = f"{area} t{t}"
            ann_rdm_dict[label] = rdm_condensed
            area_keys.append(label)

        ann_keys_by_area[area] = area_keys
        all_ann_keys.extend(area_keys)
        print(f"  {area}: {len(area_keys)} timesteps")

    if len(all_ann_keys) == 0:
        raise ValueError("No ANN RDMs found! Check --metric and --rdm_type.")

    print(f"ANN total: {len(all_ann_keys)} RDMs")

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    run_tag = f"{model_name}__{args.metric}_{args.rdm_type}"
    out_dir = path.join(args.save_dir, run_tag)
    npz_dir = path.join(out_dir, "npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    n_monkey = len(monkey_labels)
    n_ann = len(all_ann_keys)

    # ---------------------------------------------------------
    # A) Cross-correlation: monkey timecourse x ALL ANN layers
    #    (exactly like correlate_rdm_movie_with_models in notebook)
    # ---------------------------------------------------------
    print("\n=== Correlating monkey timecourse with all ANN RDMs ===")
    full_corr = correlate_rdm_movie_with_models(
        monkey_timecourse, ann_rdm_dict, all_ann_keys
    )
    # full_corr shape: (n_monkey_times, n_ann_total)

    np.savez_compressed(
        path.join(npz_dir, "full_cross_correlation.npz"),
        cross_correlation=full_corr.astype(np.float32),
        monkey_times=monkey_times_used,
        ann_keys=np.array(all_ann_keys),
        metric=np.array(args.metric),
        rdm_type=np.array(args.rdm_type),
    )
    print(f"  Saved npz/full_cross_correlation.npz")

    # ---------------------------------------------------------
    # B) Per-area cross-correlation
    # ---------------------------------------------------------
    area_corr_data = {}
    for area in AREAS:
        if area not in ann_keys_by_area:
            continue
        area_keys = ann_keys_by_area[area]

        print(f"\n=== Correlating monkey timecourse with {area} ===")
        area_corr = correlate_rdm_movie_with_models(
            monkey_timecourse, ann_rdm_dict, area_keys
        )
        # area_corr shape: (n_monkey_times, n_area_timesteps)

        area_timesteps = [int(k.split(" t")[1]) for k in area_keys]
        area_corr_data[area] = {
            "corr": area_corr,
            "ann_timesteps": area_timesteps,
        }

        np.savez_compressed(
            path.join(npz_dir, f"{area}_cross_correlation.npz"),
            cross_correlation=area_corr.astype(np.float32),
            monkey_times=monkey_times_used,
            ann_timesteps=np.array(area_timesteps, dtype=np.int32),
        )
        print(f"  Saved npz/{area}_cross_correlation.npz")

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------
    if not args.plot_panels:
        print("\nDone (plots skipped).")
        return

    print("\n=== Generating plots ===")

    # ---- Plot 1: Big cross-correlation matrix (monkey ms × all ANN layers/timesteps) ----
    # full_corr shape: (n_monkey_times, n_ann_total)
    n_ann_total = full_corr.shape[1]
    fig_w = max(14, n_ann_total * 0.35)
    fig_h = max(4, n_monkey * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(full_corr, aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=-1, vmax=1)

    # X axis: ANN keys (area t0, area t1, ...)
    ax.set_xticks(np.arange(n_ann_total))
    ax.set_xticklabels(all_ann_keys, rotation=90, fontsize=5)
    ax.set_xlabel("ANN layer / timestep")

    # Y axis: monkey timepoints (ms)
    ax.set_yticks(np.arange(n_monkey))
    ax.set_yticklabels([f"{int(t)}ms" for t in monkey_times_used], fontsize=7)
    ax.set_ylabel("Monkey time (ms)")

    # Separator lines between ANN areas
    offset = 0
    for area in AREAS:
        if area in ann_keys_by_area:
            offset += len(ann_keys_by_area[area])
            if offset < n_ann_total:
                ax.axvline(x=offset - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(f"Cross-correlation: Monkey × ANN\n{model_name}  ({args.metric}, {args.rdm_type})", fontsize=11)
    plt.tight_layout()

    big_plot_path = path.join(out_dir, "big_cross_correlation.png")
    plt.savefig(big_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {big_plot_path}")

    # ---- Plot 2: Per-area cross-correlation (monkey ms × area timesteps) ----
    available_areas = [a for a in AREAS if a in area_corr_data]
    n_areas = len(available_areas)

    if n_areas > 0:
        fig, axes = plt.subplots(1, n_areas, figsize=(4 * n_areas, 4.5))
        if n_areas == 1:
            axes = [axes]

        for ax, area in zip(axes, available_areas):
            info = area_corr_data[area]
            cross = info["corr"]         # (n_monkey_times, n_area_timesteps)
            a_ts = info["ann_timesteps"]

            im = ax.imshow(cross, aspect="auto", interpolation="nearest",
                           cmap="RdBu_r", vmin=-1, vmax=1)

            # X axis: ANN timesteps
            ax.set_xticks(np.arange(len(a_ts)))
            ax.set_xticklabels([f"t{t}" for t in a_ts], rotation=90, fontsize=7)
            ax.set_xlabel("ANN timestep")

            # Y axis: monkey times
            ax.set_yticks(np.arange(n_monkey))
            ax.set_yticklabels([f"{int(t)}ms" for t in monkey_times_used], fontsize=7)
            ax.set_ylabel("Monkey time (ms)")

            ax.set_title(area, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")

        plt.suptitle(
            f"Cross-correlation per area: {model_name}\n({args.metric}, {args.rdm_type})",
            fontsize=13
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        per_area_path = path.join(out_dir, "per_area_cross_correlation.png")
        plt.savefig(per_area_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {per_area_path}")

    # ---- Plot 3: Line plots per area (one line per ANN timestep) ----
    if len(area_corr_data) > 0:
        for area in [a for a in AREAS if a in area_corr_data]:
            info = area_corr_data[area]
            cross = info["corr"]
            a_ts = info["ann_timesteps"]

            fig, ax = plt.subplots(figsize=(8, 4))
            for j, ann_t in enumerate(a_ts):
                ax.plot(monkey_times_used, cross[:, j], label=f"t{ann_t}",
                        alpha=0.8, linewidth=1.5)

            ax.set_xlabel("Monkey time (ms)")
            ax.set_ylabel("Correlation")
            ax.set_title(f"{area} – ANN timesteps vs monkey RDM timecourse")
            ax.legend(fontsize=7, ncol=max(1, len(a_ts) // 4), loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            line_path = path.join(out_dir, f"{area}_correlation_curves.png")
            plt.savefig(line_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] {line_path}")

        # ---- Plot 4: Summary – best ANN match per area ----
        fig, ax = plt.subplots(figsize=(10, 5))
        for area in [a for a in AREAS if a in area_corr_data]:
            info = area_corr_data[area]
            cross = info["corr"]

            best_corr = np.max(cross, axis=1)

            color = AREA_COLORS.get(area, None)
            ax.plot(monkey_times_used, best_corr, label=area, linewidth=2, color=color)

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

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
