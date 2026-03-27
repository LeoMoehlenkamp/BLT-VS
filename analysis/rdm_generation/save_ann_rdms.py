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

# exakt die Metrik verwenden, die auch in der eigentlichen Monkey-Pipeline genutzt wurde
DISTANCE_METRIC = "cosine"


def reorder_features_to_original_index(features, indices):
    """
    features: (N, C)
    indices: (N,)
    Stellt die Originalreihenfolge der Stimuli wieder her:
    danach gehört Zeile i zu Stimulus i.
    """
    reordered = np.empty_like(features)
    reordered[indices] = features
    return reordered


def compute_rdm_from_features(features, metric=DISTANCE_METRIC):
    """
    features: (N, C)
    returns:
        rdm_condensed: condensed pdist vector
        rdm_square: (N, N)
    """
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
    parser.add_argument("--monkey_processed_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="analysis_outputs/ann_rdms")
    parser.add_argument("--plot_panels", type=int, default=1)
    parser.add_argument("--panel_area", type=str, default="V4")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading ANN features: {args.features_path}")
    ann_data = np.load(args.features_path, allow_pickle=True)

    print(f"Loading monkey processed data: {args.monkey_processed_path}")
    monkey_data = np.load(args.monkey_processed_path, allow_pickle=True)

    # WICHTIG: sort_idx direkt aus Monkey-Datei übernehmen
    if "sort_idx" not in monkey_data:
        raise ValueError("Monkey processed file does not contain 'sort_idx'.")
    sort_idx = monkey_data["sort_idx"].astype(int)

    if "indices" not in ann_data:
        raise ValueError("ANN feature file does not contain 'indices'.")
    indices = ann_data["indices"].astype(int)

    print(f"indices shape: {indices.shape}")
    print(f"indices first 10: {indices[:10]}")
    print(f"sort_idx shape: {sort_idx.shape}")
    print(f"sort_idx first 10: {sort_idx[:10]}")
    print(f"Using distance metric: {DISTANCE_METRIC}")

    model_name = path.basename(args.features_path).replace("_features.npz", "")
    save_base = path.join(args.save_dir, f"{model_name}_ann_rdms")

    all_save_dict = {
        "sort_idx": sort_idx.astype(np.int32),
        "indices": indices.astype(np.int32),
        "distance_metric": np.array(DISTANCE_METRIC),
    }

    panel_rdms = []
    panel_titles = []

    for area in AREAS:
        timesteps = extract_available_timesteps(ann_data, area)

        if len(timesteps) == 0:
            print(f"No timesteps found for area {area}, skipping.")
            continue

        print(f"\nProcessing area {area} with timesteps: {timesteps}")

        rdms_unsorted_all = []
        rdms_sorted_all = []
        rdms_ranked_sorted_all = []
        saved_timesteps = []

        for t in timesteps:
            key = f"{area}_t{t}"
            features = ann_data[key]

            if features is None:
                print(f"{key} is None, skipping.")
                continue

            print(f"\n{key}: feature shape = {features.shape}")

            if features.shape[0] != len(indices):
                raise ValueError(
                    f"{key}: number of rows in features ({features.shape[0]}) "
                    f"does not match indices length ({len(indices)})"
                )

            if np.isnan(features).any():
                print(f"{key}: contains NaNs, skipping.")
                continue

            # 1) Originale Stimulus-Reihenfolge wiederherstellen
            features_ordered = reorder_features_to_original_index(features, indices)

            # 2) RDM berechnen
            rdm_condensed, rdm_square = compute_rdm_from_features(
                features_ordered,
                metric=DISTANCE_METRIC
            )

            # 3) EXAKT wie Monkey sortieren
            rdm_sorted = rdm_square[sort_idx][:, sort_idx]

            # 4) ranked Version analog zur Monkey-Pipeline
            rdm_ranked = rankdata(rdm_condensed)
            rdm_ranked_square = squareform(rdm_ranked)
            rdm_ranked_sorted = rdm_ranked_square[sort_idx][:, sort_idx]

            if len(saved_timesteps) == 0:
                quick_sanity_check(rdm_square, name=f"{area}_t{t}_unsorted")
                quick_sanity_check(rdm_sorted, name=f"{area}_t{t}_sorted")

            rdms_unsorted_all.append(rdm_square.astype(np.float32))
            rdms_sorted_all.append(rdm_sorted.astype(np.float32))
            rdms_ranked_sorted_all.append(rdm_ranked_sorted.astype(np.float32))
            saved_timesteps.append(t)

            all_save_dict[f"{area}_t{t}_rdm_unsorted"] = rdm_square.astype(np.float32)
            all_save_dict[f"{area}_t{t}_rdm_sorted"] = rdm_sorted.astype(np.float32)
            all_save_dict[f"{area}_t{t}_rdm_ranked_sorted"] = rdm_ranked_sorted.astype(np.float32)

            if area == args.panel_area:
                panel_rdms.append(rdm_sorted.astype(np.float32))
                panel_titles.append(f"{area} t{t}")

        if len(saved_timesteps) > 0:
            all_save_dict[f"{area}_timesteps"] = np.array(saved_timesteps, dtype=np.int32)
            all_save_dict[f"{area}_rdms_unsorted"] = np.array(rdms_unsorted_all, dtype=np.float32)
            all_save_dict[f"{area}_rdms_sorted"] = np.array(rdms_sorted_all, dtype=np.float32)
            all_save_dict[f"{area}_rdms_ranked_sorted"] = np.array(rdms_ranked_sorted_all, dtype=np.float32)

    np.savez_compressed(save_base + ".npz", **all_save_dict)
    print(f"\nSaved ANN RDMs to: {save_base + '.npz'}")

    if args.plot_panels and len(panel_rdms) > 0:
        n_panels = len(panel_rdms)
        full_width = FULL_PANEL_SIZE[0]
        panel_size = full_width / max(n_panels, 1)

        plt.figure(figsize=(full_width, panel_size))

        for i in range(n_panels):
            plt.subplot(1, n_panels, i + 1)
            plt.imshow(panel_rdms[i], rasterized=True)
            plt.gca().axis("off")
            plt.title(panel_titles[i])

        panel_path = save_base + f"_{args.panel_area}_panel.svg"
        plt.savefig(panel_path, dpi=800, bbox_inches="tight")
        plt.close()
        print(f"Saved panel plot to: {panel_path}")


if __name__ == "__main__":
    main()