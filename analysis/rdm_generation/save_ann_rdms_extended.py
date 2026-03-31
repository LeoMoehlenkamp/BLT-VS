import os
from os import path
import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


FULL_PANEL_SIZE = (24, 6)
AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]


def reorder_features_to_original_index(features, indices):
    reordered = np.empty_like(features)
    reordered[indices] = features
    return reordered


def compute_rdm_from_features(features, metric):
    rdm_condensed = pdist(features, metric=metric)
    rdm_square = squareform(rdm_condensed)
    return rdm_condensed, rdm_square


def quick_sanity_check(rdm, name="RDM"):
    print(f"\nSanity check for {name}")
    print(f"shape: {rdm.shape}")
    print(f"min: {np.min(rdm):.6f}")
    print(f"max: {np.max(rdm):.6f}")
    print(f"mean: {np.mean(rdm):.6f}")
    print(f"has NaN: {np.isnan(rdm).any()}")
    print(f"symmetric: {np.allclose(rdm, rdm.T, atol=1e-6)}")
    print(f"diag ~ 0: {np.allclose(np.diag(rdm), 0, atol=1e-6)}")


def extract_available_timesteps(npz_file, area):
    timesteps = []
    prefix = f"{area}_t"

    for key in npz_file.files:
        if key.startswith(prefix) and "_rdm_" not in key:
            t = int(key.split("_t")[1])
            timesteps.append(t)

    return sorted(timesteps)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--monkey_processed_path", type=str, default="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua_processed.npz")
    parser.add_argument("--save_dir", type=str, default="analysis_outputs/ann_rdms")

    # 🔥 NEU
    parser.add_argument("--metric", type=str, default=None,
                        help="Distance metric (e.g. cosine, euclidean)")
    parser.add_argument("--rdm_type", type=str, default=None,
                        choices=["raw", "ranked"],
                        help="Use raw or ranked RDM")

    parser.add_argument("--plot_panels", type=int, default=1)

    args = parser.parse_args()

    # 🔥 DEFAULT LOGIC
    if args.metric is None:
        args.metric = "cosine"

    if args.rdm_type is None:
        if args.metric == "cosine":
            args.rdm_type = "ranked"
        else:
            args.rdm_type = "raw"

    print(f"\nUsing metric: {args.metric}")
    print(f"Using RDM type: {args.rdm_type}")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading ANN features: {args.features_path}")
    ann_data = np.load(args.features_path, allow_pickle=True)

    print(f"Loading monkey processed data: {args.monkey_processed_path}")
    monkey_data = np.load(args.monkey_processed_path, allow_pickle=True)

    if "sort_idx" not in monkey_data:
        raise ValueError("Monkey processed file does not contain 'sort_idx'.")
    sort_idx = monkey_data["sort_idx"].astype(int)

    if "indices" not in ann_data:
        raise ValueError("ANN feature file does not contain 'indices'.")
    indices = ann_data["indices"].astype(int)

    model_name = path.basename(args.features_path).replace("_features.npz", "")
    save_base = path.join(args.save_dir, f"{model_name}_ann_rdms")

    all_save_dict = {
        "sort_idx": sort_idx.astype(np.int32),
        "indices": indices.astype(np.int32),
        "distance_metric": np.array(args.metric),
        "rdm_type": np.array(args.rdm_type),
    }

    panel_data = {area: {"rdms": [], "titles": []} for area in AREAS}

    for area in AREAS:
        timesteps = extract_available_timesteps(ann_data, area)

        if len(timesteps) == 0:
            continue

        print(f"\nProcessing area {area} with timesteps: {timesteps}")

        saved_timesteps = []

        for t in timesteps:
            key = f"{area}_t{t}"
            features = ann_data[key]

            if features is None or np.isnan(features).any():
                continue

            features_ordered = reorder_features_to_original_index(features, indices)

            # 🔥 RDM berechnen
            rdm_condensed, rdm_square = compute_rdm_from_features(
                features_ordered,
                metric=args.metric
            )

            rdm_sorted = rdm_square[sort_idx][:, sort_idx]

            # ranked
            rdm_ranked = rankdata(rdm_condensed)
            rdm_ranked_square = squareform(rdm_ranked)
            rdm_ranked_sorted = rdm_ranked_square[sort_idx][:, sort_idx]

            # 🔥 Auswahl
            if args.rdm_type == "raw":
                rdm_final = rdm_sorted
            else:
                rdm_final = rdm_ranked_sorted

            if len(saved_timesteps) == 0:
                quick_sanity_check(rdm_final, name=f"{area}_t{t}_{args.metric}_{args.rdm_type}")

            # 🔥 speichern
            all_save_dict[f"{area}_t{t}_rdm_{args.metric}_{args.rdm_type}"] = rdm_final.astype(np.float32)

            # optional: beide speichern
            all_save_dict[f"{area}_t{t}_rdm_{args.metric}_raw"] = rdm_sorted.astype(np.float32)
            all_save_dict[f"{area}_t{t}_rdm_{args.metric}_ranked"] = rdm_ranked_sorted.astype(np.float32)

            panel_data[area]["rdms"].append(rdm_final.astype(np.float32))
            panel_data[area]["titles"].append(f"{area} t{t}")

            saved_timesteps.append(t)

        if len(saved_timesteps) > 0:
            all_save_dict[f"{area}_timesteps"] = np.array(saved_timesteps, dtype=np.int32)

    np.savez_compressed(save_base + ".npz", **all_save_dict)
    print(f"\nSaved ANN RDMs to: {save_base + '.npz'}")

    # ---------------------------
    # PLOTS
    # ---------------------------
    if args.plot_panels:
        for area in AREAS:
            area_rdms = panel_data[area]["rdms"]
            area_titles = panel_data[area]["titles"]

            if len(area_rdms) == 0:
                continue

            n_panels = len(area_rdms)
            full_width = FULL_PANEL_SIZE[0]
            panel_size = full_width / max(n_panels, 1)

            plt.figure(figsize=(full_width, panel_size))

            for i in range(n_panels):
                plt.subplot(1, n_panels, i + 1)
                plt.imshow(area_rdms[i], rasterized=True)
                plt.gca().axis("off")
                plt.title(area_titles[i])

            panel_path = save_base + f"_{area}_panel.svg"
            plt.savefig(panel_path, dpi=800, bbox_inches="tight")
            plt.close()

        print("Saved panel plots.")


if __name__ == "__main__":
    main()