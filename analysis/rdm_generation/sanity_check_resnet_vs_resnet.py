"""
Sanity check: correlate RDMs from two pretrained ResNet variants layer-by-layer.

Loads two .npz files produced by extract_resnet_features_and_rdms.py and
computes a layer × layer second-order correlation matrix.

Usage:
  python sanity_check_resnet_vs_resnet.py \
      --rdm_path_a <resnet50_rdms.npz> \
      --rdm_path_b <resnet101_rdms.npz> \
      [--metric cosine] [--rdm_type ranked] [--save_dir ...]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, cdist


STAGES = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]

STAGE_COLORS = {
    "conv1":  "#1f77b4",
    "layer1": "#ff7f0e",
    "layer2": "#2ca02c",
    "layer3": "#d62728",
    "layer4": "#9467bd",
    "fc":     "#8c564b",
}


def get_stage(layer_name):
    """Map a full layer name (e.g. layer2.1.conv3) to its stage."""
    for stage in ["layer4", "layer3", "layer2", "layer1", "conv1", "fc"]:
        if layer_name.startswith(stage):
            return stage
    return layer_name


def get_layer_color(layer_name):
    return STAGE_COLORS.get(get_stage(layer_name), "#333333")


def load_resnet_rdms(npz_path, metric, rdm_type):
    """Load RDM vectors and layer names from a ResNet .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    variant = str(data["resnet_variant"]) if "resnet_variant" in data else "resnet"
    layers = list(data["layers"]) if "layers" in data else []

    vectors = []
    labels = []
    for layer in layers:
        key = f"{layer}_rdm_{metric}_{rdm_type}"
        if key not in data:
            print(f"  Warning: {key} not found, skipping")
            continue
        rdm = data[key].astype(np.float64)
        if rdm.ndim == 2:
            rdm = squareform(rdm)
        vectors.append(rdm)
        labels.append(layer)

    return np.array(vectors), labels, variant


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: correlate two ResNet RDM sets"
    )
    parser.add_argument("--rdm_path_a", type=str, required=True,
                        help="Path to first ResNet RDMs (.npz)")
    parser.add_argument("--rdm_path_b", type=str, required=True,
                        help="Path to second ResNet RDMs (.npz)")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"])
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/sanity_check_resnet")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load both
    print(f"Loading A: {args.rdm_path_a}")
    vecs_a, labels_a, variant_a = load_resnet_rdms(args.rdm_path_a, args.metric, args.rdm_type)
    print(f"  {variant_a}: {len(labels_a)} layers, vector length {vecs_a.shape[1]}")

    print(f"Loading B: {args.rdm_path_b}")
    vecs_b, labels_b, variant_b = load_resnet_rdms(args.rdm_path_b, args.metric, args.rdm_type)
    print(f"  {variant_b}: {len(labels_b)} layers, vector length {vecs_b.shape[1]}")

    # Correlate: A layers × B layers
    corr_matrix = 1 - cdist(vecs_a, vecs_b, metric="correlation")
    print(f"\nCorrelation matrix shape: {corr_matrix.shape}")
    print(f"  min={corr_matrix.min():.4f}  max={corr_matrix.max():.4f}  mean={corr_matrix.mean():.4f}")

    # Save npz
    tag = f"{variant_a}_vs_{variant_b}__{args.metric}_{args.rdm_type}"
    out_dir = os.path.join(args.save_dir, tag)
    npz_dir = os.path.join(out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    npz_path = os.path.join(npz_dir, f"{tag}.npz")
    np.savez_compressed(
        npz_path,
        similarity_matrix=corr_matrix.astype(np.float32),
        row_labels=np.array(labels_a),
        col_labels=np.array(labels_b),
        variant_a=np.array(variant_a),
        variant_b=np.array(variant_b),
    )
    print(f"Saved: {npz_path}")

    # ---------------------------------------------------------
    # Group ResNet B layers by stage
    # ---------------------------------------------------------
    stage_to_cols = {}
    for col_idx, lbl in enumerate(labels_b):
        stage = get_stage(lbl)
        stage_to_cols.setdefault(stage, []).append((col_idx, lbl))

    # Preserve canonical stage order
    available_stages = [s for s in STAGES if s in stage_to_cols]

    # ---------------------------------------------------------
    # Plot 1: Full heatmap with stage boundaries
    # ---------------------------------------------------------
    n_rows, n_cols = corr_matrix.shape
    fig_w = max(12, n_cols * 0.35)
    fig_h = max(6, n_rows * 0.25)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr_matrix, aspect="auto", interpolation="nearest",
                   cmap="Reds", vmin=np.min(corr_matrix), vmax=np.max(corr_matrix))

    # Stage boundaries
    offset = 0
    boundaries = []
    for stage in available_stages:
        offset += len(stage_to_cols[stage])
        boundaries.append(offset)
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

    # Best layer per stage dots
    block_offset = 0
    for stage in available_stages:
        block_size = len(stage_to_cols[stage])
        for row_idx in range(n_rows):
            block_corrs = corr_matrix[row_idx, block_offset:block_offset + block_size]
            best_col = block_offset + np.argmax(block_corrs)
            ax.scatter(best_col, row_idx, color="white", edgecolors="black",
                       linewidths=0.5, s=40, zorder=5)
        block_offset += block_size

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels_a, fontsize=max(4, min(8, 200 // max(n_rows, 1))))
    ax.set_ylabel(variant_a)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(labels_b, rotation=90, fontsize=5)
    ax.set_xlabel(variant_b)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(f"Second-Order Similarity + Best Layer per Stage\n"
                 f"{variant_a} vs {variant_b} ({args.metric}, {args.rdm_type})")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"full_heatmap_best_layer.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig_path}")

    # ---------------------------------------------------------
    # Plot 2: summary_best_corr_per_area
    # For each ResNet A layer, best correlation per ResNet B stage
    # ---------------------------------------------------------
    n_layers = len(labels_a)
    fig_w = max(10, n_layers * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x_pos = np.arange(n_layers)
    bar_width = 0.8 / max(len(available_stages), 1)

    for i, stage in enumerate(available_stages):
        col_indices = [ci for ci, _ in stage_to_cols[stage]]
        best_corr = np.max(corr_matrix[:, col_indices], axis=1)
        color = STAGE_COLORS.get(stage, "#333333")
        ax.bar(x_pos + i * bar_width, best_corr, bar_width,
               label=stage, color=color, alpha=0.85)

    ax.set_xticks(x_pos + bar_width * len(available_stages) / 2)
    ax.set_xticklabels(labels_a, fontsize=max(5, min(8, 200 // max(n_layers, 1))),
                       rotation=90)
    ax.set_xlabel(f"{variant_a} layer")
    ax.set_ylabel(f"Best correlation with {variant_b}")
    ax.set_title(f"Best {variant_b} match per stage — {variant_a} vs {variant_b}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    summary_path = os.path.join(out_dir, "summary_best_corr_per_area.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {summary_path}")

    # ---------------------------------------------------------
    # Plot 3: summary_best_overall_corr
    # For each ResNet A layer, best correlation with any ResNet B layer
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    best_overall = np.max(corr_matrix, axis=1)
    best_labels = [labels_b[j] for j in np.argmax(corr_matrix, axis=1)]

    bar_colors = [get_layer_color(l) for l in labels_a]
    bars = ax.bar(labels_a, best_overall, color=bar_colors, alpha=0.85)
    ax.set_xlabel(f"{variant_a} layer")
    ax.set_ylabel(f"Best correlation with any {variant_b} layer")
    ax.set_title(f"Best overall {variant_a} vs {variant_b}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=90, fontsize=max(5, min(8, 200 // max(n_layers, 1))))

    for bar, label in zip(bars, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=5, rotation=45)

    plt.tight_layout()
    overall_path = os.path.join(out_dir, "summary_best_overall_corr.png")
    plt.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overall_path}")

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
