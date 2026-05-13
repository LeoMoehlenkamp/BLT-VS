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

    # Save
    tag = f"{variant_a}_vs_{variant_b}__{args.metric}_{args.rdm_type}"
    npz_path = os.path.join(args.save_dir, f"{tag}.npz")
    np.savez_compressed(
        npz_path,
        similarity_matrix=corr_matrix.astype(np.float32),
        row_labels=np.array(labels_a),
        col_labels=np.array(labels_b),
        variant_a=np.array(variant_a),
        variant_b=np.array(variant_b),
    )
    print(f"Saved: {npz_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(labels_b) * 0.35),
                                     max(6, len(labels_a) * 0.35)))
    im = ax.imshow(corr_matrix, aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(labels_a)))
    ax.set_yticklabels(labels_a, fontsize=6)
    ax.set_ylabel(variant_a)
    ax.set_xticks(np.arange(len(labels_b)))
    ax.set_xticklabels(labels_b, rotation=90, fontsize=6)
    ax.set_xlabel(variant_b)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Correlation")
    ax.set_title(f"Sanity check: {variant_a} vs {variant_b} ({args.metric}, {args.rdm_type})")
    plt.tight_layout()

    fig_path = os.path.join(args.save_dir, f"{tag}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
