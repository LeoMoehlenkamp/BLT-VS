"""
Generate ResNet RDMs and panel plots — analogous to save_ann_rdms_extended.py.

Reads pre-extracted features from the ResNet .npz (from
extract_resnet_features_and_rdms.py), computes RDMs for the requested metric,
and saves UNSORTED RDMs (compatible with second_order_rdms_resnet_vs_monkey.py)
while generating SORTED panel plots for visual inspection.

Usage:
  python save_resnet_rdms_extended.py \
      --features_path analysis_outputs/resnet_rdms/resnet50_rdms.npz \
      --monkey_processed_path <monkey_processed.npz> \
      --save_dir analysis_outputs/resnet_rdms/resnet50_cosine_ranked \
      --metric cosine --rdm_type ranked --plot_panels 1
"""

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
STAGE_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


def get_stage(layer_name):
    for stage in STAGE_ORDER:
        if layer_name.startswith(stage) or layer_name == stage:
            return stage
    return "other"


def compute_rdm_from_features(features, metric):
    rdm_condensed = pdist(features, metric=metric)
    rdm_square = squareform(rdm_condensed)
    return rdm_condensed, rdm_square


def quick_sanity_check(rdm, name="RDM"):
    print(f"\nSanity check for {name}")
    print(f"  shape: {rdm.shape}")
    print(f"  min:   {np.min(rdm):.6f}")
    print(f"  max:   {np.max(rdm):.6f}")
    print(f"  mean:  {np.mean(rdm):.6f}")
    print(f"  NaN:   {np.isnan(rdm).any()}")
    print(f"  sym:   {np.allclose(rdm, rdm.T, atol=1e-6)}")
    print(f"  diag0: {np.allclose(np.diag(rdm), 0, atol=1e-6)}")


def get_color_limits(rdms):
    if len(rdms) == 0:
        return None, None
    offdiag_vals = []
    for rdm in rdms:
        mask = ~np.eye(rdm.shape[0], dtype=bool)
        offdiag_vals.append(rdm[mask])
    offdiag_vals = np.concatenate(offdiag_vals)
    return float(np.min(offdiag_vals)), float(np.max(offdiag_vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to ResNet npz from extract_resnet_features_and_rdms.py")
    parser.add_argument("--monkey_processed_path", type=str,
                        default="analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua/"
                                "monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-"
                                "arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-"
                                "baseline_0-standardize_1-metric_correlation-"
                                "neural_mua_processed.npz")
    parser.add_argument("--save_dir", type=str, default="analysis_outputs/resnet_rdms")
    parser.add_argument("--metric", type=str, default=None,
                        help="Distance metric (e.g. cosine, euclidean)")
    parser.add_argument("--rdm_type", type=str, default=None,
                        choices=["raw", "ranked"],
                        help="Use raw or ranked RDM")
    parser.add_argument("--plot_panels", type=int, default=1)
    args = parser.parse_args()

    # Default logic (matches save_ann_rdms_extended.py)
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

    # ---------------------------
    # Load data
    # ---------------------------
    print(f"Loading ResNet data: {args.features_path}")
    resnet_data = np.load(args.features_path, allow_pickle=True)

    print(f"Loading monkey processed data: {args.monkey_processed_path}")
    monkey_data = np.load(args.monkey_processed_path, allow_pickle=True)

    if "sort_idx" not in monkey_data:
        raise ValueError("Monkey processed file does not contain 'sort_idx'.")
    sort_idx = monkey_data["sort_idx"].astype(int)

    # Get layer names
    if "layers" in resnet_data:
        layer_names = list(resnet_data["layers"])
    else:
        layer_names = sorted([
            k.replace("_features", "")
            for k in resnet_data.files
            if k.endswith("_features")
        ])

    resnet_variant = str(resnet_data["resnet_variant"]) if "resnet_variant" in resnet_data else "resnet"

    save_base = path.join(args.save_dir, f"{resnet_variant}_resnet_rdms")

    all_save_dict = {
        "sort_idx": sort_idx.astype(np.int32),
        "resnet_variant": np.array(resnet_variant),
        "layers": np.array(layer_names),
        "distance_metric": np.array(args.metric),
        "rdm_type": np.array(args.rdm_type),
    }

    # Panel data grouped by stage (for plotting sorted RDMs)
    panel_data = {}

    first_done = False
    saved_layers = []

    for layer in layer_names:
        feat_key = f"{layer}_features"
        if feat_key not in resnet_data:
            print(f"  Warning: {feat_key} not found, skipping")
            continue

        features = resnet_data[feat_key]
        if features is None or np.isnan(features).any():
            print(f"  Warning: {layer} has NaN features, skipping")
            continue

        print(f"  Processing {layer} ({features.shape})")

        # Compute RDM (features are already in original stimulus order)
        rdm_condensed, rdm_square = compute_rdm_from_features(
            features, metric=args.metric
        )

        # Ranked version
        rdm_ranked = rankdata(rdm_condensed)
        rdm_ranked_square = squareform(rdm_ranked)

        # Select final (unsorted) for saving
        if args.rdm_type == "raw":
            rdm_final_unsorted = rdm_square
        else:
            rdm_final_unsorted = rdm_ranked_square

        if not first_done:
            quick_sanity_check(rdm_final_unsorted, name=f"{layer}_{args.metric}_{args.rdm_type}")
            first_done = True

        # Save UNSORTED RDMs (compatible with second_order_rdms_resnet_vs_monkey.py)
        all_save_dict[f"{layer}_rdm_{args.metric}_{args.rdm_type}"] = rdm_final_unsorted.astype(np.float32)
        all_save_dict[f"{layer}_rdm_{args.metric}_raw"] = rdm_square.astype(np.float32)
        all_save_dict[f"{layer}_rdm_{args.metric}_ranked"] = rdm_ranked_square.astype(np.float32)

        saved_layers.append(layer)

        # For visualization: sort by category
        rdm_sorted = rdm_square[sort_idx][:, sort_idx]
        rdm_ranked_sorted = rdm_ranked_square[sort_idx][:, sort_idx]
        rdm_final_sorted = rdm_sorted if args.rdm_type == "raw" else rdm_ranked_sorted

        stage = get_stage(layer)
        if stage not in panel_data:
            panel_data[stage] = {"rdms": [], "titles": []}
        panel_data[stage]["rdms"].append(rdm_final_sorted.astype(np.float32))
        panel_data[stage]["titles"].append(layer)

    # Update layers list to only include actually saved layers
    all_save_dict["layers"] = np.array(saved_layers)

    np.savez_compressed(save_base + ".npz", **all_save_dict)
    print(f"\nSaved ResNet RDMs to: {save_base}.npz")

    # ---------------------------
    # PLOTS (using SORTED RDMs)
    # ---------------------------
    if not args.plot_panels:
        return

    # Color limits per stage (for raw RDMs)
    stage_color_limits = {}
    if args.rdm_type == "raw":
        for stage in STAGE_ORDER:
            if stage in panel_data:
                stage_color_limits[stage] = get_color_limits(panel_data[stage]["rdms"])

    # ---------------------------
    # Per-stage panel plots
    # ---------------------------
    for stage in STAGE_ORDER:
        if stage not in panel_data:
            continue

        rdms = panel_data[stage]["rdms"]
        titles = panel_data[stage]["titles"]
        n_panels = len(rdms)
        full_width = FULL_PANEL_SIZE[0]
        panel_size = full_width / max(n_panels, 1)

        plt.figure(figsize=(full_width, panel_size))
        vmin, vmax = stage_color_limits.get(stage, (None, None))

        for i in range(n_panels):
            plt.subplot(1, n_panels, i + 1)
            if args.rdm_type == "raw":
                im = plt.imshow(rdms[i], rasterized=True, vmin=vmin, vmax=vmax)
                plt.colorbar(im, fraction=0.046, pad=0.04)
            else:
                im = plt.imshow(rdms[i], rasterized=True)
            plt.gca().axis("off")
            # e.g. "layer2.0.conv1" → "0.conv1"
            parts = titles[i].split(".")
            short_title = ".".join(parts[1:]) if len(parts) > 2 else titles[i]
            plt.title(short_title, fontsize=7)

        panel_path = save_base + f"_{stage}_panel.svg"
        plt.savefig(panel_path, dpi=800, bbox_inches="tight")
        plt.close()
        print(f"[PANEL SAVED] {stage}: {panel_path}")

    # ---------------------------
    # Combined overview: stages (rows) × sub-layers (cols)
    # ---------------------------
    active_stages = [s for s in STAGE_ORDER if s in panel_data]
    n_rows = len(active_stages)
    max_cols = max(len(panel_data[s]["rdms"]) for s in active_stages) if active_stages else 0

    if n_rows > 0 and max_cols > 0:
        fig, axes = plt.subplots(
            n_rows, max_cols,
            figsize=(2.6 * max_cols, 2.4 * n_rows),
            squeeze=False
        )

        for row, stage in enumerate(active_stages):
            rdms = panel_data[stage]["rdms"]
            titles = panel_data[stage]["titles"]
            vmin, vmax = stage_color_limits.get(stage, (None, None))

            for col in range(max_cols):
                ax = axes[row, col]
                if col < len(rdms):
                    if args.rdm_type == "raw":
                        ax.imshow(rdms[col], interpolation="nearest", vmin=vmin, vmax=vmax)
                    else:
                        ax.imshow(rdms[col], interpolation="nearest")
                    short = ".".join(titles[col].split(".")[1:]) if len(titles[col].split(".")) > 2 else titles[col]
                    ax.set_title(short, fontsize=6)
                ax.axis("off")

        # Stage labels on the left
        plt.tight_layout(rect=[0.08, 0, 1, 0.97])
        for row, stage in enumerate(active_stages):
            pos = axes[row, 0].get_position()
            y_center = (pos.y0 + pos.y1) / 2
            fig.text(0.04, y_center, stage, va="center", ha="right", fontsize=10)

        plt.suptitle(f"{resnet_variant} ({args.metric}, {args.rdm_type})", fontsize=14)

        combined_path = save_base + "_combined_panel.png"
        plt.savefig(combined_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[COMBINED PANEL SAVED] {combined_path}")

    print("Panel plots done.")


if __name__ == "__main__":
    main()
