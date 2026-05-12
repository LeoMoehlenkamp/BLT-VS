"""
Unique-variance analysis: BLT-VS vs ResNet → Monkey neural RDMs.

For each monkey timepoint, this script asks:
  How much variance in the monkey RDM is uniquely explained by
  BLT-VS (not shared with ResNet) and vice versa?

Strategy: For each model, select the best-matching layer/area+timestep
per monkey timepoint (or use a fixed one), then run unique_variance_per_model.

Supports:
  - BLT-VS ANN RDMs from save_ann_rdms_extended.py (.npz)
  - ResNet RDMs from either TIMM .pkl or own extraction .npz
  - Monkey RDMs from .pkl (raw) with on-the-fly sorting + ranking

Usage:
  python analysis/rdm_generation/unique_variance_blt_vs_resnet.py \
      --ann_rdm_path  <blt_vs_rdms.npz> \
      --resnet_rdm_npz <resnet_rdms.npz> \
      --monkey_pkl_path <monkey.pkl> \
      --stimulus_csv <stimulus_information.csv> \
      --layer_selection best \
      --blt_area V4 \
      --save_dir analysis_outputs/unique_variance
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
from sklearn.linear_model import LinearRegression


AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

DEFAULT_MONKEY_PKL = (
    "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_mua_minithings/"
    "monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3"
    "-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16"
    "-baseline_0-standardize_1-metric_correlation-neural_mua.pkl"
)
DEFAULT_STIMULUS_CSV = (
    "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
)


# ============================================================
# Core analysis
# ============================================================

def unique_variance_per_model(models, rdm, zscore=False, fit_intercept=False,
                              positive=True):
    """
    Compute unique variance explained by each predictor (model RDM) in `models`
    for a single target `rdm`, using semi-partial R².

    Parameters
    ----------
    models : (n_predictors, n_pairs)
    rdm    : (n_pairs,)

    Returns
    -------
    full_r2         : float
    partial_r2s     : (n_predictors,)  — R² without each predictor
    unique_vars     : (n_predictors,)  — full_r2 - partial_r2s
    """
    assert len(rdm.shape) == 1
    assert models.shape[1] == len(rdm)

    y = rdm[:, None]
    x = models.T

    if zscore:
        y = (y - y.mean()) / y.std()
        x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)

    reg = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    reg.fit(x, y)
    full_r2 = reg.score(x, y)

    partial_r2s = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        xsub = np.delete(x, i, axis=1)
        reg.fit(xsub, y)
        partial_r2s[i] = reg.score(xsub, y)

    unique_vars = full_r2 - partial_r2s
    return full_r2, partial_r2s, unique_vars


# ============================================================
# Helpers (consistent with second_order_rdms_ann_vs_monkey.py)
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


def extract_matching_keys(npz_file, area, metric, rdm_type):
    """Return sorted (timestep, key) pairs for a given area/metric/rdm_type."""
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


# ============================================================
# Data loading
# ============================================================

def load_monkey_rdms(args, stimulus_csv):
    """Load and pre-process monkey RDMs (sort + optional rank)."""
    print(f"Loading monkey RDMs: {args.monkey_pkl_path}")
    with open(args.monkey_pkl_path, "rb") as f:
        monkey_rdm_data = pickle.load(f)

    # Use the label column from the monkey data config (e.g. "filenames")
    # to get sort indices that preserve all stimuli (not collapsed to categories)
    label_col = monkey_rdm_data["data_cfg"]["labels"]
    print(f"  Monkey label column: {label_col}")
    sort_idx = get_rdm_design_sort_indices(stimulus_csv, reduce_to_column=label_col)

    monkey_time = np.array(monkey_rdm_data["time"])
    monkey_rdms_raw = monkey_rdm_data["rdms"]
    t_select = np.arange(args.t_start, args.t_end, args.t_step)

    timecourse = []
    times_used = []

    for t in t_select:
        matches = np.where(monkey_time == t)[0]
        if len(matches) == 0:
            print(f"  Warning: monkey time {t} not found, skipping")
            continue

        rdm = monkey_rdms_raw[matches[0]].astype(np.float64)
        if args.rdm_type == "ranked":
            rdm = rankdata(rdm)

        rdm = squareform(rdm)
        rdm = rdm[sort_idx][:, sort_idx]
        rdm = squareform(rdm)

        timecourse.append(rdm)
        times_used.append(t)

    timecourse = np.array(timecourse, dtype=np.float64)
    times_used = np.array(times_used, dtype=np.int32)
    print(f"Monkey: {len(times_used)} timepoints, {timecourse.shape[1]} pairs")
    return timecourse, times_used, monkey_rdm_data


def load_ann_rdms(args, ann_data):
    """Load BLT-VS ANN RDMs from .npz, return dict label→condensed and key list."""
    rdm_dict = {}
    keys_by_area = {}
    all_keys = []

    for area in AREAS:
        matches = extract_matching_keys(ann_data, area, args.metric, args.rdm_type)
        if not matches:
            continue

        area_keys = []
        for t, npz_key in matches:
            arr = ann_data[npz_key].astype(np.float64)
            if arr.ndim == 2:
                condensed = squareform(arr)
            else:
                condensed = arr
            label = f"{area} t{t}"
            rdm_dict[label] = condensed
            area_keys.append(label)

        keys_by_area[area] = area_keys
        all_keys.extend(area_keys)
        print(f"  BLT-VS {area}: {len(area_keys)} timesteps")

    return rdm_dict, keys_by_area, all_keys


def load_resnet_rdms(args):
    """Load ResNet RDMs from pkl or npz."""
    rdm_dict = {}
    all_keys = []

    if args.resnet_rdm_pkl is not None:
        print(f"Loading ResNet RDMs from pkl: {args.resnet_rdm_pkl}")
        with open(args.resnet_rdm_pkl, "rb") as f:
            data = pickle.load(f)
        all_keys = data["selected_nodes"]
        rdm_dict = data["rdms"]
        variant = (path.basename(args.resnet_rdm_pkl).split("-")[1]
                   if "-" in path.basename(args.resnet_rdm_pkl) else "resnet")
    else:
        print(f"Loading ResNet RDMs from npz: {args.resnet_rdm_npz}")
        data = np.load(args.resnet_rdm_npz, allow_pickle=True)
        variant = (str(data["resnet_variant"])
                   if "resnet_variant" in data else "resnet")
        layers = list(data["layers"]) if "layers" in data else []
        for layer in layers:
            key = f"{layer}_rdm_{args.metric}_{args.rdm_type}"
            if key not in data:
                print(f"  Warning: {key} not found, skipping")
                continue
            arr = data[key].astype(np.float64)
            if arr.ndim == 2:
                arr = squareform(arr)
            rdm_dict[layer] = arr
            all_keys.append(layer)

    print(f"ResNet ({variant}): {len(all_keys)} layers")
    return rdm_dict, all_keys, variant


# ============================================================
# Layer selection strategies
# ============================================================

def select_best_rdm(target, rdm_dict, keys):
    """Select the key from `keys` whose RDM is most correlated with `target`."""
    stack = np.array([rdm_dict[k] for k in keys])
    corrs = 1 - cdist(target[None, :], stack, metric="correlation").squeeze()
    best_idx = np.argmax(corrs)
    return keys[best_idx], corrs[best_idx]


def select_fixed_rdm(rdm_dict, keys, fixed_label):
    """Select a specific RDM by label. Raises if not found."""
    if fixed_label not in rdm_dict:
        available = [k for k in keys if fixed_label.lower() in k.lower()]
        if len(available) == 1:
            fixed_label = available[0]
        else:
            raise KeyError(
                f"'{fixed_label}' not found. Available: {keys[:20]}..."
            )
    return fixed_label


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unique-variance analysis: BLT-VS vs ResNet → Monkey"
    )

    # --- Data paths ---
    parser.add_argument("--ann_rdm_path", type=str, required=True,
                        help="BLT-VS .npz from save_ann_rdms_extended.py")
    parser.add_argument("--resnet_rdm_npz", type=str, default=None,
                        help="ResNet .npz from your own extraction")
    parser.add_argument("--resnet_rdm_pkl", type=str, default=None,
                        help="ResNet .pkl from TIMM (takes priority)")
    parser.add_argument("--monkey_pkl_path", type=str, default=DEFAULT_MONKEY_PKL)
    parser.add_argument("--stimulus_csv", type=str, default=DEFAULT_STIMULUS_CSV)

    # --- RDM settings ---
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"])

    # --- Monkey time window ---
    parser.add_argument("--t_start", type=int, default=0)
    parser.add_argument("--t_end", type=int, default=400)
    parser.add_argument("--t_step", type=int, default=10)

    # --- Layer selection ---
    parser.add_argument("--layer_selection", type=str, default="best",
                        choices=["best", "fixed"],
                        help="'best' = pick best-matching layer per timepoint; "
                             "'fixed' = use a single layer/area specified below")
    parser.add_argument("--blt_area", type=str, default=None,
                        help="BLT-VS areas to include (comma-separated, "
                             "e.g. 'V4' or 'V1,V4,LOC'). Default: all areas.")
    parser.add_argument("--blt_timestep", type=int, default=None,
                        help="Fixed BLT-VS timestep (used with --layer_selection fixed)")
    parser.add_argument("--resnet_layer", type=str, default=None,
                        help="Fixed ResNet layer (used with --layer_selection fixed)")

    # --- Regression settings ---
    parser.add_argument("--zscore", type=int, default=1)
    parser.add_argument("--positive", type=int, default=1,
                        help="Constrain regression coefficients to be positive")

    # --- Output ---
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/unique_variance")
    parser.add_argument("--plot", type=int, default=1)

    args = parser.parse_args()

    if args.resnet_rdm_pkl is None and args.resnet_rdm_npz is None:
        parser.error("Provide --resnet_rdm_npz or --resnet_rdm_pkl")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    monkey_tc, monkey_times, monkey_meta = load_monkey_rdms(args, args.stimulus_csv)

    print(f"\nLoading BLT-VS ANN RDMs: {args.ann_rdm_path}")
    ann_data = np.load(args.ann_rdm_path, allow_pickle=True)
    ann_rdm_dict, ann_keys_by_area, all_ann_keys = load_ann_rdms(args, ann_data)

    resnet_rdm_dict, all_resnet_keys, resnet_variant = load_resnet_rdms(args)

    # Filter BLT-VS areas if requested
    if args.blt_area is not None:
        selected_areas = [a.strip() for a in args.blt_area.split(",")]
        filtered_keys = []
        for area in selected_areas:
            if area in ann_keys_by_area:
                filtered_keys.extend(ann_keys_by_area[area])
            else:
                print(f"  Warning: area '{area}' not found in ANN RDMs")
        if not filtered_keys:
            raise ValueError(f"No ANN RDMs for areas: {selected_areas}")
        blt_candidate_keys = filtered_keys
        area_tag = "_".join(selected_areas)
    else:
        blt_candidate_keys = all_ann_keys
        area_tag = "all"

    print(f"\nBLT-VS candidate RDMs: {len(blt_candidate_keys)}")
    print(f"ResNet candidate RDMs: {len(all_resnet_keys)}")

    # ---------------------------------------------------------
    # 2. Run unique variance analysis per monkey timepoint
    # ---------------------------------------------------------
    n_times = len(monkey_times)
    results = {
        "monkey_times": monkey_times,
        "full_r2": np.empty(n_times),
        "blt_unique": np.empty(n_times),
        "resnet_unique": np.empty(n_times),
        "shared": np.empty(n_times),
        "blt_partial_r2": np.empty(n_times),
        "resnet_partial_r2": np.empty(n_times),
        "blt_selected": [],
        "resnet_selected": [],
        "blt_corr_at_selection": np.empty(n_times),
        "resnet_corr_at_selection": np.empty(n_times),
    }

    print(f"\n{'='*60}")
    print(f"Running unique variance analysis: {n_times} monkey timepoints")
    print(f"Layer selection: {args.layer_selection}")
    print(f"Z-score: {bool(args.zscore)}, Positive: {bool(args.positive)}")
    print(f"{'='*60}")

    for i, t in enumerate(monkey_times):
        target = monkey_tc[i]  # (n_pairs,)

        # --- Select BLT-VS RDM ---
        if args.layer_selection == "best":
            blt_key, blt_corr = select_best_rdm(target, ann_rdm_dict,
                                                 blt_candidate_keys)
        else:
            if args.blt_timestep is None:
                raise ValueError("--blt_timestep required with --layer_selection fixed")
            if args.blt_area is None:
                raise ValueError("--blt_area required with --layer_selection fixed "
                                 "(specify a single area)")
            blt_key = select_fixed_rdm(
                ann_rdm_dict, blt_candidate_keys,
                f"{selected_areas[0]} t{args.blt_timestep}"
            )
            blt_corr = float(
                1 - cdist(target[None, :],
                          ann_rdm_dict[blt_key][None, :],
                          metric="correlation").squeeze()
            )

        # --- Select ResNet RDM ---
        if args.layer_selection == "best":
            rn_key, rn_corr = select_best_rdm(target, resnet_rdm_dict,
                                               all_resnet_keys)
        else:
            if args.resnet_layer is None:
                raise ValueError("--resnet_layer required with --layer_selection fixed")
            rn_key = select_fixed_rdm(resnet_rdm_dict, all_resnet_keys,
                                      args.resnet_layer)
            rn_corr = float(
                1 - cdist(target[None, :],
                          resnet_rdm_dict[rn_key][None, :],
                          metric="correlation").squeeze()
            )

        # --- Build predictor matrix ---
        models = np.stack([ann_rdm_dict[blt_key],
                           resnet_rdm_dict[rn_key]], axis=0)  # (2, n_pairs)

        full_r2, partial_r2s, unique_vars = unique_variance_per_model(
            models, target,
            zscore=bool(args.zscore),
            positive=bool(args.positive),
        )

        shared = full_r2 - unique_vars.sum()

        results["full_r2"][i] = full_r2
        results["blt_unique"][i] = unique_vars[0]
        results["resnet_unique"][i] = unique_vars[1]
        results["shared"][i] = shared
        results["blt_partial_r2"][i] = partial_r2s[0]
        results["resnet_partial_r2"][i] = partial_r2s[1]
        results["blt_selected"].append(blt_key)
        results["resnet_selected"].append(rn_key)
        results["blt_corr_at_selection"][i] = blt_corr
        results["resnet_corr_at_selection"][i] = rn_corr

        print(f"  t={t:4d}ms | BLT={blt_key:15s} (r={blt_corr:.3f}) | "
              f"RN={rn_key:25s} (r={rn_corr:.3f}) | "
              f"R²={full_r2:.4f} | BLT_u={unique_vars[0]:.4f} | "
              f"RN_u={unique_vars[1]:.4f} | shared={shared:.4f}")

    # ---------------------------------------------------------
    # 3. Save results
    # ---------------------------------------------------------
    model_name = path.basename(args.ann_rdm_path).replace("_ann_rdms.npz", "").replace(".npz", "")
    run_tag = f"{model_name}_vs_{resnet_variant}__{args.layer_selection}_{area_tag}"
    out_dir = path.join(args.save_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = path.join(out_dir, "unique_variance_results.npz")
    np.savez_compressed(
        npz_path,
        monkey_times=results["monkey_times"],
        full_r2=results["full_r2"],
        blt_unique=results["blt_unique"],
        resnet_unique=results["resnet_unique"],
        shared=results["shared"],
        blt_partial_r2=results["blt_partial_r2"],
        resnet_partial_r2=results["resnet_partial_r2"],
        blt_selected=np.array(results["blt_selected"]),
        resnet_selected=np.array(results["resnet_selected"]),
        blt_corr_at_selection=results["blt_corr_at_selection"],
        resnet_corr_at_selection=results["resnet_corr_at_selection"],
        layer_selection=args.layer_selection,
        zscore=args.zscore,
        positive=args.positive,
        metric=args.metric,
        rdm_type=args.rdm_type,
    )
    print(f"\nSaved results: {npz_path}")

    # ---------------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------------
    if not args.plot:
        print("Done. Plots skipped.")
        return

    times = results["monkey_times"]

    # --- Plot A: Stacked area chart (unique + shared) ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(times, 0, results["blt_unique"],
                    alpha=0.6, label="BLT-VS unique", color="#2ca02c")
    ax.fill_between(times, results["blt_unique"],
                    results["blt_unique"] + results["shared"],
                    alpha=0.4, label="Shared", color="#7f7f7f")
    ax.fill_between(times, results["blt_unique"] + results["shared"],
                    results["full_r2"],
                    alpha=0.6, label="ResNet unique", color="#d62728")

    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("Variance explained (R²)")
    ax.set_title(f"Unique Variance: {model_name} vs {resnet_variant} → Monkey\n"
                 f"(selection={args.layer_selection}, areas={area_tag})")
    ax.legend(loc="upper right")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(path.join(out_dir, "unique_variance_stacked.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] unique_variance_stacked.png")

    # --- Plot B: Line plot of each component ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, results["full_r2"], "k-", linewidth=2, label="Full model R²")
    ax.plot(times, results["blt_unique"], "-", color="#2ca02c", linewidth=1.5,
            label="BLT-VS unique")
    ax.plot(times, results["resnet_unique"], "-", color="#d62728", linewidth=1.5,
            label="ResNet unique")
    ax.plot(times, results["shared"], "--", color="#7f7f7f", linewidth=1.5,
            label="Shared")

    ax.set_xlabel("Monkey time (ms)")
    ax.set_ylabel("R²")
    ax.set_title(f"Unique Variance Components: {model_name} vs {resnet_variant}")
    ax.legend(loc="upper right")
    ax.set_xlim(times[0], times[-1])
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(path.join(out_dir, "unique_variance_lines.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] unique_variance_lines.png")

    # --- Plot C: Bar chart of selected layers over time ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    # BLT-VS selections
    blt_labels = results["blt_selected"]
    unique_blt = sorted(set(blt_labels))
    blt_y = [unique_blt.index(l) for l in blt_labels]
    axes[0].scatter(times, blt_y, c=results["blt_corr_at_selection"],
                    cmap="Greens", edgecolors="black", linewidths=0.5, s=50)
    axes[0].set_yticks(range(len(unique_blt)))
    axes[0].set_yticklabels(unique_blt, fontsize=7)
    axes[0].set_ylabel("BLT-VS")
    axes[0].set_title("Selected layers per monkey timepoint (color = correlation)")

    # ResNet selections
    rn_labels = results["resnet_selected"]
    unique_rn = sorted(set(rn_labels))
    rn_y = [unique_rn.index(l) for l in rn_labels]
    axes[1].scatter(times, rn_y, c=results["resnet_corr_at_selection"],
                    cmap="Reds", edgecolors="black", linewidths=0.5, s=50)
    axes[1].set_yticks(range(len(unique_rn)))
    axes[1].set_yticklabels(unique_rn, fontsize=7)
    axes[1].set_ylabel("ResNet")
    axes[1].set_xlabel("Monkey time (ms)")

    plt.tight_layout()
    fig.savefig(path.join(out_dir, "selected_layers.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] selected_layers.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
