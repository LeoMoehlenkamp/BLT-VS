"""
Rectangular second-order RDM analysis: ResNet layers vs. BLT-VS model RDMs.

Analogous to second_order_rdms_ann_vs_monkey.py, but with ResNet layers
replacing monkey timepoints:
  - rows    = ResNet layers
  - columns = BLT-VS areas + timesteps

Usage:
  python second_order_rdms_ann_vs_resnet.py \
      --ann_rdm_path <path_to_blt_vs_ann_rdms.npz> \
      --resnet_rdm_path <path_to_resnet_rdms.npz> \
      [--metric cosine] [--rdm_type ranked]
"""

import os
from os import path
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, cdist
from scipy.stats import rankdata


BLT_AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

AREA_COLORS = {
    "Retina": "#1f77b4",
    "LGN":    "#ff7f0e",
    "V1":     "#2ca02c",
    "V2":     "#d62728",
    "V3":     "#9467bd",
    "V4":     "#8c564b",
    "LOC":    "#e377c2",
}

RESNET_LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]

# Colors: assign per stage, sub-layers within a stage share the same color
STAGE_COLORS = {
    "conv1":   "#1f77b4",
    "layer1":  "#ff7f0e",
    "layer2":  "#2ca02c",
    "layer3":  "#d62728",
    "layer4":  "#9467bd",
    "avgpool": "#8c564b",
}


def get_layer_color(layer_name):
    """Get color for a layer based on its stage (e.g. layer2.1.conv3 -> layer2 color)."""
    for stage in ["layer4", "layer3", "layer2", "layer1", "avgpool", "conv1"]:
        if layer_name.startswith(stage):
            return STAGE_COLORS[stage]
    return "#333333"


# ============================================================
# Helpers
# ============================================================

def correlate_rdms(reference_rdms, target_rdms, target_keys):
    """
    Correlate reference RDMs with target RDMs.

    reference_rdms: (n_ref, n_pairs) — e.g. ResNet layers
    target_rdms:    dict label -> condensed RDM vector
    target_keys:    list of labels

    returns: (n_ref, n_target)
    """
    targets = np.array([target_rdms[key] for key in target_keys], dtype=np.float64)
    return 1 - cdist(reference_rdms, targets, metric="correlation")


def extract_matching_keys(npz_file, area, metric, rdm_type):
    """
    Return sorted (timestep, key) pairs for a given area/metric/rdm_type.
    Supports new format: {area}_t{T}_rdm_{metric}_{rdm_type}
    and old format: {area}_t{T}_rdm_ranked_sorted / {area}_t{T}_rdm_sorted
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
    matrix, row_labels, col_labels, save_path, title,
    xlabel, ylabel, vmin=-1, vmax=1, cmap="Reds",
    figsize=None, x_group_boundaries=None,
):
    """Plot a rectangular heatmap."""
    n_rows, n_cols = matrix.shape

    if figsize is None:
        fig_w = max(12, n_cols * 0.35)
        fig_h = max(4, n_rows * 0.6)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
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
    """Build vertical separator positions between BLT-VS areas."""
    boundaries = []
    offset = 0
    for area in ordered_areas:
        if area in ann_keys_by_area:
            offset += len(ann_keys_by_area[area])
            boundaries.append(offset)
    if boundaries:
        boundaries = boundaries[:-1]
    return boundaries


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rectangular second-order RDM: ResNet layers vs BLT-VS"
    )
    parser.add_argument("--ann_rdm_path", type=str, required=True,
                        help="Path to BLT-VS RDMs (.npz from save_ann_rdms_extended.py)")
    parser.add_argument("--resnet_rdm_path", type=str, required=True,
                        help="Path to ResNet RDMs (.npz from extract_resnet_features_and_rdms.py)")
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/second_order_ann_vs_resnet")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric used in first-order RDMs")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"],
                        help="Use raw or ranked first-order RDMs")
    parser.add_argument("--plot_panels", type=int, default=1)
    parser.add_argument("--display_name", type=str, default="",
                        help="Override the model name shown in plot titles")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load ResNet RDMs
    # ---------------------------------------------------------
    print(f"Loading ResNet RDMs: {args.resnet_rdm_path}")
    resnet_data = np.load(args.resnet_rdm_path, allow_pickle=True)

    resnet_variant = str(resnet_data["resnet_variant"]) if "resnet_variant" in resnet_data else "resnet"
    available_layers = list(resnet_data["layers"]) if "layers" in resnet_data else RESNET_LAYERS

    resnet_rdm_vectors = []
    resnet_labels = []

    for layer in available_layers:
        key = f"{layer}_rdm_{args.metric}_{args.rdm_type}"
        if key not in resnet_data:
            # Try without sorting suffix
            alt_key = f"{layer}_rdm_{args.metric}_{args.rdm_type}_sorted"
            if alt_key in resnet_data:
                key = alt_key
            else:
                print(f"  Warning: {key} not found in ResNet npz, skipping")
                continue

        rdm = resnet_data[key].astype(np.float64)
        if rdm.ndim == 2:
            rdm_vec = squareform(rdm)
        else:
            rdm_vec = rdm

        resnet_rdm_vectors.append(rdm_vec)
        resnet_labels.append(layer)

    if len(resnet_rdm_vectors) == 0:
        raise ValueError("No ResNet RDMs found. Check --metric / --rdm_type.")

    resnet_timecourse = np.array(resnet_rdm_vectors, dtype=np.float64)
    print(f"ResNet layers loaded: {resnet_labels}")
    print(f"ResNet RDM matrix shape: {resnet_timecourse.shape}")

    # ---------------------------------------------------------
    # Load BLT-VS (ANN) RDMs
    # ---------------------------------------------------------
    print(f"\nLoading BLT-VS RDMs: {args.ann_rdm_path}")
    ann_data = np.load(args.ann_rdm_path, allow_pickle=True)

    model_name = path.basename(args.ann_rdm_path).replace("_ann_rdms.npz", "").replace(".npz", "")

    ann_rdm_dict = {}
    ann_keys_by_area = {}
    all_ann_keys = []

    for area in BLT_AREAS:
        matches = extract_matching_keys(ann_data, area, args.metric, args.rdm_type)
        if not matches:
            print(f"  {area}: no matches found")
            continue

        area_keys = []
        for t, npz_key in matches:
            arr = ann_data[npz_key].astype(np.float64)
            if arr.ndim == 2:
                rdm_condensed = squareform(arr)
            elif arr.ndim == 1:
                rdm_condensed = arr
            else:
                raise ValueError(f"Unexpected shape for {npz_key}: {arr.shape}")

            label = f"{area} t{t}"
            ann_rdm_dict[label] = rdm_condensed
            area_keys.append(label)

        ann_keys_by_area[area] = area_keys
        all_ann_keys.extend(area_keys)
        print(f"  {area}: {len(area_keys)} timesteps")

    if not all_ann_keys:
        raise ValueError("No BLT-VS RDMs found. Check --metric / --rdm_type.")

    print(f"BLT-VS total RDMs: {len(all_ann_keys)}")

    # ---------------------------------------------------------
    # Prepare output dirs
    # ---------------------------------------------------------
    run_tag = f"{model_name}_vs_{resnet_variant}__{args.metric}_{args.rdm_type}"
    out_dir = path.join(args.save_dir, run_tag)
    npz_dir = path.join(out_dir, "npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    # ---------------------------------------------------------
    # A) Full rectangular: ResNet layers × ALL BLT-VS RDMs
    # ---------------------------------------------------------
    print("\n=== Correlating ResNet layers with all BLT-VS RDMs ===")
    full_corr = correlate_rdms(resnet_timecourse, ann_rdm_dict, all_ann_keys)
    print(f"  Correlation matrix shape: {full_corr.shape}")

    np.savez_compressed(
        path.join(npz_dir, "rectangular_resnet_vs_blt.npz"),
        similarity_matrix=full_corr.astype(np.float32),
        row_labels=np.array(resnet_labels),
        col_labels=np.array(all_ann_keys),
        metric=np.array(args.metric),
        rdm_type=np.array(args.rdm_type),
    )
    print("  Saved npz/rectangular_resnet_vs_blt.npz")

    # ---------------------------------------------------------
    # B) Per BLT-VS area: ResNet layers × area timesteps
    # ---------------------------------------------------------
    area_corr_data = {}

    for area in BLT_AREAS:
        if area not in ann_keys_by_area:
            continue

        area_keys = ann_keys_by_area[area]
        print(f"\n=== Correlating ResNet layers with {area} ===")

        area_corr = correlate_rdms(resnet_timecourse, ann_rdm_dict, area_keys)
        area_timesteps = [int(k.split(" t")[1]) for k in area_keys]

        area_corr_data[area] = {
            "corr": area_corr,
            "ann_timesteps": area_timesteps,
            "ann_keys": area_keys,
        }

        np.savez_compressed(
            path.join(npz_dir, f"{area}_rectangular_cross_correlation.npz"),
            similarity_matrix=area_corr.astype(np.float32),
            row_labels=np.array(resnet_labels),
            col_labels=np.array([f"t{t}" for t in area_timesteps]),
            ann_timesteps=np.array(area_timesteps, dtype=np.int32),
        )
        print(f"  Saved npz/{area}_rectangular_cross_correlation.npz")

    # ---------------------------------------------------------
    # Stop if no plots
    # ---------------------------------------------------------
    if not args.plot_panels:
        print("\nDone. Plots skipped.")
        return

    print("\n=== Generating plots ===")

    x_boundaries = build_x_boundaries(ann_keys_by_area, BLT_AREAS)

    # ---------------------------------------------------------
    # Plot 1: Full heatmap — ResNet layers × all BLT-VS
    # ---------------------------------------------------------
    vmin_full = np.min(full_corr)
    vmax_full = np.max(full_corr)

    plot_rectangular_matrix(
        matrix=full_corr,
        row_labels=resnet_labels,
        col_labels=all_ann_keys,
        save_path=path.join(out_dir, "full_resnet_vs_blt_heatmap.png"),
        title=f"Second-Order Similarity: {resnet_variant} layers vs BLT-VS\n{model_name} ({args.metric}, {args.rdm_type})",
        xlabel="BLT-VS area / timestep",
        ylabel="ResNet layer",
        vmin=vmin_full, vmax=vmax_full,
        cmap="Reds",
        x_group_boundaries=x_boundaries,
    )
    print(f"  [SAVED] full_resnet_vs_blt_heatmap.png")

    # ---------------------------------------------------------
    # Plot 1b: Full heatmap + best timestep dots
    # ---------------------------------------------------------
    n_rows, n_cols = full_corr.shape
    fig_w = max(12, n_cols * 0.35)
    fig_h = max(6, n_rows * 0.25)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(full_corr, aspect="auto", interpolation="nearest",
                   cmap="Reds", vmin=vmin_full, vmax=vmax_full)

    # Best timestep per ResNet layer per BLT-VS area
    block_offset = 0
    for area in BLT_AREAS:
        if area not in ann_keys_by_area:
            continue
        block_size = len(ann_keys_by_area[area])
        for row_idx in range(n_rows):
            block_corrs = full_corr[row_idx, block_offset:block_offset + block_size]
            best_local = np.argmax(block_corrs)
            best_col = block_offset + best_local
            ax.scatter(best_col, row_idx, color="white", edgecolors="black",
                       linewidths=0.5, s=40, zorder=5)
        block_offset += block_size

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(resnet_labels, fontsize=max(4, min(8, 200 // max(n_rows, 1))))
    ax.set_ylabel("ResNet layer")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(all_ann_keys, rotation=90, fontsize=5)
    ax.set_xlabel("BLT-VS area / timestep")

    if x_boundaries:
        for b in x_boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(
        f"Second-Order Similarity + Best Timestep per Area\n"
        f"{resnet_variant} vs {model_name} ({args.metric}, {args.rdm_type})",
        fontsize=11,
    )
    plt.tight_layout()
    overlay_path = path.join(out_dir, "full_resnet_vs_blt_best_timestep.png")
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overlay_path}")

    # ---------------------------------------------------------
    # Plot 2: Per-area heatmaps
    # ---------------------------------------------------------
    available_areas = [a for a in BLT_AREAS if a in area_corr_data]

    for area in available_areas:
        info = area_corr_data[area]
        cross = info["corr"]
        a_ts = info["ann_timesteps"]

        area_vmin = np.min(cross)
        area_vmax = np.max(cross)

        save_path = path.join(out_dir, f"{area}_rectangular_cross_correlation.png")
        plot_rectangular_matrix(
            matrix=cross,
            row_labels=resnet_labels,
            col_labels=[f"t{t}" for t in a_ts],
            save_path=save_path,
            title=f"{area}: {resnet_variant} vs BLT-VS timesteps – {model_name}",
            xlabel="BLT-VS timestep",
            ylabel="ResNet layer",
            vmin=area_vmin, vmax=area_vmax,
            cmap="Reds",
            figsize=(max(5, len(a_ts) * 0.65), max(4, len(resnet_labels) * 0.5)),
        )
        print(f"  [SAVED] {save_path}")

    # ---------------------------------------------------------
    # Plot 3: Per-area line plots
    # x-axis = BLT-VS timestep, one line per ResNet layer
    # ---------------------------------------------------------
    for area in available_areas:
        info = area_corr_data[area]
        cross = info["corr"]  # (n_resnet_layers, n_area_timesteps)
        a_ts = info["ann_timesteps"]

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, rn_layer in enumerate(resnet_labels):
            color = get_layer_color(rn_layer)
            ax.plot(a_ts, cross[j, :], label=rn_layer, alpha=0.7,
                    linewidth=1.2, color=color)

        ax.set_xlabel("BLT-VS timestep")
        ax.set_ylabel("Correlation")
        ax.set_title(f"{area} – ResNet layers vs BLT-VS timesteps – {model_name}")
        ax.legend(fontsize=5, ncol=max(1, len(resnet_labels) // 8), loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        line_path = path.join(out_dir, f"{area}_correlation_curves.png")
        plt.savefig(line_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {line_path}")

    # ---------------------------------------------------------
    # Plot 4: Summary — best BLT-VS match per ResNet layer, per area
    # For each ResNet layer, take best BLT-VS timestep within each area
    # ---------------------------------------------------------
    if available_areas:
        n_layers = len(resnet_labels)
        fig_w = max(10, n_layers * 0.4)

        fig, ax = plt.subplots(figsize=(fig_w, 5))

        x_pos = np.arange(n_layers)
        bar_width = 0.8 / max(len(available_areas), 1)

        for i, area in enumerate(available_areas):
            info = area_corr_data[area]
            cross = info["corr"]
            best_corr = np.max(cross, axis=1)

            color = AREA_COLORS.get(area, None)
            ax.bar(x_pos + i * bar_width, best_corr, bar_width,
                   label=area, color=color, alpha=0.85)

        ax.set_xticks(x_pos + bar_width * len(available_areas) / 2)
        ax.set_xticklabels(resnet_labels, fontsize=max(5, min(8, 200 // max(n_layers, 1))),
                           rotation=90)
        ax.set_xlabel("ResNet layer")
        ax.set_ylabel("Best correlation with BLT-VS")
        ax.set_title(f"Best BLT-VS match per area – {model_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        summary_path = path.join(out_dir, "summary_best_corr_per_area.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {summary_path}")

    # ---------------------------------------------------------
    # Plot 5: Overall best correlation per ResNet layer
    # ---------------------------------------------------------
    n_layers = len(resnet_labels)
    fig_w = max(10, n_layers * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    best_overall = np.max(full_corr, axis=1)
    best_labels = [all_ann_keys[j] for j in np.argmax(full_corr, axis=1)]

    bar_colors = [get_layer_color(l) for l in resnet_labels]
    bars = ax.bar(resnet_labels, best_overall, color=bar_colors, alpha=0.85)
    ax.set_xlabel("ResNet layer")
    ax.set_ylabel("Best correlation with any BLT-VS RDM")
    if args.display_name:
        _short_name = args.display_name
    else:
        _parts = model_name.split("__")
        _short_name = _parts[3] if len(_parts) > 3 else model_name
        _short_name = _short_name.replace("bn-", "BN").replace("-", "_")
    ax.set_title(f"Best overall {resnet_variant} vs BLT-VS – {_short_name}", pad=25)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=90, fontsize=max(5, min(8, 200 // max(n_layers, 1))))

    # Annotate bars with the best-matching BLT-VS label
    for bar, label in zip(bars, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=5, rotation=45)

    plt.tight_layout()
    overall_path = path.join(out_dir, "summary_best_overall_corr.png")
    plt.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overall_path}")

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
