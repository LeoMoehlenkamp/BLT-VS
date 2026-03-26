import os
from os import path
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from scipy.spatial.distance import squareform


def get_rdm_design_sort_indices(stimulus_csv, reduce_to_column="category", return_values=False):
    stim_info = pd.read_csv(stimulus_csv)

    stim_info_sorted = stim_info.sort_values(
        [
            "animate",
            "body_parts",
            "human",
            "mammal",
            "non_mammal",
            "inanimate",
            "natural",
            "food",
            "fruit",
            "vegetable",
            "other_food",
            "plants",
            "other_natural",
            "artificial",
            "artificial_small",
            "tools",
            "artificial_small_other",
            "artificial_large",
            "furniture",
            "vehicles",
            "outside_large",
            "cat_id",
        ],
        ascending=False,
    )

    stim_info_select = stim_info_sorted[reduce_to_column]
    stim_info_select = stim_info_select.drop_duplicates()

    indices = stim_info_select.index.values
    stim_info_select_allcols = stim_info.iloc[indices]
    sort_idx = rankdata(stim_info_select.index.values).astype(int) - 1

    if not return_values:
        return sort_idx
    else:
        return sort_idx, stim_info_select.values, stim_info_select_allcols


def to_square_rdm(rdm):
    rdm = np.asarray(rdm)

    if rdm.ndim == 2:
        return rdm
    elif rdm.ndim == 1:
        return squareform(rdm)
    else:
        raise ValueError(f"Unsupported RDM shape: {rdm.shape}")


def notebook_rank_transform(rdm_square):
    ranked = rankdata(rdm_square)
    ranked_square = squareform(ranked)
    return ranked_square


def save_single_plot(rdm_square, save_path, title=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(rdm_square, rasterized=True)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_panel_plot(rdms_ranked_sorted, time, save_path, t_select=None):
    if t_select is None:
        t_select = time

    selected_indices = []
    selected_times = []

    for t in t_select:
        matches = np.where(time == t)[0]
        if len(matches) == 0:
            print(f"Warning: requested time {t} not found, skipping.")
            continue
        selected_indices.append(matches[0])
        selected_times.append(t)

    n_panels = len(selected_indices)
    if n_panels == 0:
        print("No valid timepoints found for panel plot. Skipping.")
        return

    full_width = max(2 * n_panels, 12)
    panel_height = max(full_width / n_panels, 2)

    plt.figure(figsize=(full_width, panel_height))

    for i, idx in enumerate(selected_indices):
        rdm = rdms_ranked_sorted[idx]
        plt.subplot(1, n_panels, i + 1)
        plt.imshow(rdm, rasterized=True)
        plt.gca().axis("off")
        plt.title(f"{int(selected_times[i])}ms")

    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.close()


def main(args):
    if not path.exists(args.rdm_path):
        raise FileNotFoundError(f"RDM file not found: {args.rdm_path}")
    if not path.exists(args.stimulus_csv):
        raise FileNotFoundError(f"Stimulus CSV not found: {args.stimulus_csv}")

    run_name = path.splitext(path.basename(args.rdm_path))[0]
    out_dir = path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading RDM file: {args.rdm_path}")
    with open(args.rdm_path, "rb") as f:
        rdm_data = pickle.load(f)

    if "rdms" not in rdm_data:
        raise KeyError("Key 'rdms' not found in pickle file.")
    if "time" not in rdm_data:
        raise KeyError("Key 'time' not found in pickle file.")
    if "data_cfg" not in rdm_data or "labels" not in rdm_data["data_cfg"]:
        raise KeyError("Expected rdm_data['data_cfg']['labels'].")

    rdms_raw = rdm_data["rdms"]
    time = np.asarray(rdm_data["time"])
    reduce_to_column = rdm_data["data_cfg"]["labels"]

    print(f"Number of timepoints / RDMs: {len(time)}")
    print(f"Sorting column: {reduce_to_column}")

    sort_idx = get_rdm_design_sort_indices(
        stimulus_csv=args.stimulus_csv,
        reduce_to_column=reduce_to_column,
        return_values=False
    )

    rdms_square = []
    rdms_sorted = []
    rdms_ranked_sorted = []

    for i, t in enumerate(time):
        rdm_square = to_square_rdm(rdms_raw[i])

        if rdm_square.shape[0] != len(sort_idx):
            raise ValueError(
                f"RDM shape {rdm_square.shape} does not match sort_idx length {len(sort_idx)} at time {t}."
            )

        rdm_sorted = rdm_square[sort_idx][:, sort_idx]

        rdm_ranked_sorted = notebook_rank_transform(rdm_square)
        rdm_ranked_sorted = rdm_ranked_sorted[sort_idx][:, sort_idx]

        rdms_square.append(rdm_square)
        rdms_sorted.append(rdm_sorted)
        rdms_ranked_sorted.append(rdm_ranked_sorted)

        if not args.no_images:
            save_single_plot(
                rdm_ranked_sorted,
                path.join(out_dir, f"rdm_{int(t):03d}ms.png"),
                title=f"{int(t)} ms"
            )

    rdms_square = np.array(rdms_square)
    rdms_sorted = np.array(rdms_sorted)
    rdms_ranked_sorted = np.array(rdms_ranked_sorted)

    np.savez_compressed(
        path.join(out_dir, "monkey_rdms.npz"),
        time=time,
        sort_idx=sort_idx,
        rdms_square=rdms_square,
        rdms_sorted=rdms_sorted,
        rdms_ranked_sorted=rdms_ranked_sorted,
    )

    np.save(path.join(out_dir, "time.npy"), time)
    np.save(path.join(out_dir, "sort_idx.npy"), sort_idx)
    np.save(path.join(out_dir, "rdms_square.npy"), rdms_square)
    np.save(path.join(out_dir, "rdms_sorted.npy"), rdms_sorted)
    np.save(path.join(out_dir, "rdms_ranked_sorted.npy"), rdms_ranked_sorted)

    if not args.no_panel:
        save_panel_plot(
            rdms_ranked_sorted,
            time,
            path.join(out_dir, "rdm_timecourse_panel.png"),
            t_select=args.t_select
        )
        save_panel_plot(
            rdms_ranked_sorted,
            time,
            path.join(out_dir, "rdm_timecourse_panel.svg"),
            t_select=args.t_select
        )

    with open(path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Source RDM file: {args.rdm_path}\n")
        f.write(f"Stimulus CSV: {args.stimulus_csv}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Number of RDMs: {len(time)}\n")
        f.write(f"Times: {time.tolist()}\n")
        f.write(f"Sorting column: {reduce_to_column}\n")
        f.write(f"sort_idx length: {len(sort_idx)}\n")
        f.write(f"rdms_square shape: {rdms_square.shape}\n")
        f.write(f"rdms_sorted shape: {rdms_sorted.shape}\n")
        f.write(f"rdms_ranked_sorted shape: {rdms_ranked_sorted.shape}\n")

    print(f"Done. Saved outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save sorted monkey RDMs using THINGS_Drift metadata.")
    parser.add_argument("--rdm_path", type=str, required=True, help="Full path to the monkey RDM .pkl file")
    parser.add_argument("--stimulus_csv", type=str, required=True, help="Full path to stimulus_information.csv")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/monkey_rdms", help="Output directory")
    parser.add_argument("--no_images", action="store_true", help="Do not save per-timepoint PNGs")
    parser.add_argument("--no_panel", action="store_true", help="Do not save combined panel plot")
    parser.add_argument(
        "--t_select",
        type=int,
        nargs="*",
        default=None,
        help="Optional timepoints for panel plot, e.g. --t_select 0 10 20 30"
    )

    args = parser.parse_args()
    main(args)