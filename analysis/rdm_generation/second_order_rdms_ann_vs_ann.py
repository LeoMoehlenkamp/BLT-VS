"""
Second-order RDM comparison between two ANN models.

For each area, computes Spearman correlation between the RDM vectors
of Model A and Model B across all timestep combinations:
  - Cross-model time-time heatmaps (full matrix)
  - Diagonal summary (matching timesteps)
  - Summary across areas

Usage:
  python second_order_rdms_ann_vs_ann.py \
      --rdm_path_a <path_to_model_a_ann_rdms.npz> \
      --rdm_path_b <path_to_model_b_ann_rdms.npz> \
      --save_dir <output_dir> \
      --metric cosine \
      --rdm_type ranked
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
from scipy.stats import spearmanr

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


def rdm_to_vector(rdm):
    """Convert square RDM to condensed vector, or pass through if already condensed."""
    if rdm.ndim == 2:
        return squareform(rdm, checks=False)
    return rdm


def main():
    parser = argparse.ArgumentParser(
        description="Second-order RDM comparison between two ANN models"
    )
    parser.add_argument("--rdm_path_a", type=str, required=True,
                        help="Path to Model A RDMs (.npz)")
    parser.add_argument("--rdm_path_b", type=str, required=True,
                        help="Path to Model B RDMs (.npz)")
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/second_order_ann_vs_ann")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--rdm_type", type=str, default="ranked",
                        choices=["raw", "ranked"])
    parser.add_argument("--label_a", type=str, default=None,
                        help="Short label for Model A (auto-derived if omitted)")
    parser.add_argument("--label_b", type=str, default=None,
                        help="Short label for Model B (auto-derived if omitted)")
    args = parser.parse_args()

    # Derive short model labels from file paths
    label_a = args.label_a or path.basename(args.rdm_path_a).replace("_ann_rdms.npz", "")
    label_b = args.label_b or path.basename(args.rdm_path_b).replace("_ann_rdms.npz", "")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load RDMs for both models
    # ---------------------------------------------------------
    print(f"Loading Model A RDMs: {args.rdm_path_a}")
    data_a = np.load(args.rdm_path_a, allow_pickle=True)

    print(f"Loading Model B RDMs: {args.rdm_path_b}")
    data_b = np.load(args.rdm_path_b, allow_pickle=True)

    # ---------------------------------------------------------
    # Per-area cross-model time-time comparison
    # ---------------------------------------------------------
    area_results = {}

    for area in AREAS:
        matches_a = extract_matching_keys(data_a, area, args.metric, args.rdm_type)
        matches_b = extract_matching_keys(data_b, area, args.metric, args.rdm_type)

        if len(matches_a) == 0:
            print(f"  {area}: no RDMs found in Model A → skipping")
            continue
        if len(matches_b) == 0:
            print(f"  {area}: no RDMs found in Model B → skipping")
            continue

        ts_a = [t for t, _ in matches_a]
        ts_b = [t for t, _ in matches_b]

        rdms_a = np.array([
            rdm_to_vector(data_a[key].astype(np.float64)) for _, key in matches_a
        ])
        rdms_b = np.array([
            rdm_to_vector(data_b[key].astype(np.float64)) for _, key in matches_b
        ])

        print(f"\n  {area}: Model A has {len(ts_a)} timesteps, Model B has {len(ts_b)} timesteps")

        # Compute Spearman correlation between all pairs (A_ti, B_tj)
        n_a = len(ts_a)
        n_b = len(ts_b)
        cross_corr = np.zeros((n_a, n_b))

        for i in range(n_a):
            for j in range(n_b):
                rho, _ = spearmanr(rdms_a[i], rdms_b[j])
                cross_corr[i, j] = rho

        area_results[area] = {
            "cross_corr": cross_corr,
            "ts_a": ts_a,
            "ts_b": ts_b,
        }

        # Save per-area npz
        np.savez_compressed(
            path.join(args.save_dir, f"{area}_cross_model_similarity.npz"),
            cross_corr=cross_corr.astype(np.float32),
            timesteps_a=np.array(ts_a, dtype=np.int32),
            timesteps_b=np.array(ts_b, dtype=np.int32),
            label_a=np.array(label_a),
            label_b=np.array(label_b),
            metric=np.array(args.metric),
            rdm_type=np.array(args.rdm_type),
        )
        print(f"    Saved {area}_cross_model_similarity.npz")

    available_areas = [a for a in AREAS if a in area_results]

    if len(available_areas) == 0:
        print("No matching areas found between the two models.")
        return

    # =========================================================
    # PLOT 1: Per-area cross-model time-time heatmaps
    # =========================================================
    print("\n=== Generating per-area heatmaps ===")

    for area in available_areas:
        info = area_results[area]
        cross = info["cross_corr"]
        ts_a = info["ts_a"]
        ts_b = info["ts_b"]

        fig, ax = plt.subplots(figsize=(max(5, len(ts_b) * 0.55),
                                        max(4, len(ts_a) * 0.55)))
        vmin = np.min(cross)
        vmax = np.max(cross)

        im = ax.imshow(cross, aspect="auto", cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(len(ts_b)))
        ax.set_xticklabels([f"t{t}" for t in ts_b], fontsize=8)
        ax.set_yticks(range(len(ts_a)))
        ax.set_yticklabels([f"t{t}" for t in ts_a], fontsize=8)

        ax.set_xlabel(f"Model B ({label_b})")
        ax.set_ylabel(f"Model A ({label_a})")
        ax.set_title(f"{area} – Cross-Model RDM Similarity\n({args.metric}, {args.rdm_type})")

        fig.colorbar(im, ax=ax, label="Spearman ρ")
        plt.tight_layout()
        save_path = path.join(args.save_dir, f"{area}_cross_model_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {save_path}")

    # =========================================================
    # PLOT 2: Overview — all areas side by side
    # =========================================================
    n = len(available_areas)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))

    if n == 1:
        axes = [axes]

    for ax, area in zip(axes, available_areas):
        info = area_results[area]
        cross = info["cross_corr"]
        ts_a = info["ts_a"]
        ts_b = info["ts_b"]

        im = ax.imshow(cross, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")

        ax.set_xticks(range(len(ts_b)))
        ax.set_xticklabels([f"t{t}" for t in ts_b], rotation=90, fontsize=7)
        ax.set_yticks(range(len(ts_a)))
        ax.set_yticklabels([f"t{t}" for t in ts_a], fontsize=7)
        ax.set_title(area, fontsize=10)

        if ax == axes[0]:
            ax.set_ylabel(f"Model A ({label_a})")
        ax.set_xlabel(f"Model B")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Cross-Model RDM Similarity ({args.metric}, {args.rdm_type})\n"
        f"A: {label_a}  vs  B: {label_b}",
        fontsize=11,
    )
    plt.tight_layout()
    overview_path = path.join(args.save_dir, "overview_cross_model_heatmaps.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {overview_path}")

    # =========================================================
    # PLOT 3: Diagonal summary (matching timesteps)
    # =========================================================
    print("\n=== Generating diagonal summary ===")

    fig, ax = plt.subplots(figsize=(10, 5))

    for area in available_areas:
        info = area_results[area]
        cross = info["cross_corr"]
        ts_a = info["ts_a"]
        ts_b = info["ts_b"]

        # Find matching timesteps
        common_ts = sorted(set(ts_a) & set(ts_b))
        if len(common_ts) == 0:
            continue

        diag_corr = []
        for t in common_ts:
            i = ts_a.index(t)
            j = ts_b.index(t)
            diag_corr.append(cross[i, j])

        color = AREA_COLORS.get(area, None)
        ax.plot(common_ts, diag_corr, marker="o", label=area,
                linewidth=2, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Spearman ρ (same timestep)")
    ax.set_title(
        f"Cross-Model Similarity at Matching Timesteps ({args.metric}, {args.rdm_type})\n"
        f"A: {label_a}  vs  B: {label_b}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    diag_path = path.join(args.save_dir, "diagonal_similarity_per_area.png")
    plt.savefig(diag_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {diag_path}")

    # =========================================================
    # PLOT 4: Summary bar chart — mean diagonal similarity per area
    # =========================================================
    print("\n=== Generating mean similarity bar chart ===")

    area_names = []
    mean_corrs = []
    colors = []

    for area in available_areas:
        info = area_results[area]
        cross = info["cross_corr"]
        ts_a = info["ts_a"]
        ts_b = info["ts_b"]

        common_ts = sorted(set(ts_a) & set(ts_b))
        if len(common_ts) == 0:
            continue

        diag_corr = []
        for t in common_ts:
            i = ts_a.index(t)
            j = ts_b.index(t)
            diag_corr.append(cross[i, j])

        area_names.append(area)
        mean_corrs.append(np.mean(diag_corr))
        colors.append(AREA_COLORS.get(area, "#333333"))

    if len(area_names) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(area_names, mean_corrs, color=colors, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, mean_corrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Mean Spearman ρ (diagonal)")
        ax.set_title(
            f"Mean Cross-Model Similarity per Area ({args.metric}, {args.rdm_type})\n"
            f"A: {label_a}  vs  B: {label_b}"
        )
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        bar_path = path.join(args.save_dir, "mean_similarity_per_area.png")
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {bar_path}")

    # =========================================================
    # PLOT 5: Full cross-model matrix (all areas × timesteps)
    # =========================================================
    print("\n=== Generating full cross-model similarity matrix ===")

    all_labels_a = []
    all_labels_b = []
    all_vecs_a = []
    all_vecs_b = []

    for area in available_areas:
        matches_a = extract_matching_keys(data_a, area, args.metric, args.rdm_type)
        matches_b = extract_matching_keys(data_b, area, args.metric, args.rdm_type)

        for t, key in matches_a:
            all_labels_a.append(f"{area} t{t}")
            all_vecs_a.append(rdm_to_vector(data_a[key].astype(np.float64)))
        for t, key in matches_b:
            all_labels_b.append(f"{area} t{t}")
            all_vecs_b.append(rdm_to_vector(data_b[key].astype(np.float64)))

    if len(all_vecs_a) > 0 and len(all_vecs_b) > 0:
        mat_a = np.array(all_vecs_a)
        mat_b = np.array(all_vecs_b)

        # Compute full cross-correlation via 1 - correlation distance
        full_cross = 1 - cdist(mat_a, mat_b, metric="correlation")

        np.savez_compressed(
            path.join(args.save_dir, "full_cross_model_similarity.npz"),
            similarity_matrix=full_cross.astype(np.float32),
            labels_a=np.array(all_labels_a),
            labels_b=np.array(all_labels_b),
            label_a=np.array(label_a),
            label_b=np.array(label_b),
            metric=np.array(args.metric),
            rdm_type=np.array(args.rdm_type),
        )

        # Compute area boundaries for separators
        boundaries_a = []
        boundaries_b = []
        offset_a, offset_b = 0, 0
        for area in available_areas:
            na = len(extract_matching_keys(data_a, area, args.metric, args.rdm_type))
            nb = len(extract_matching_keys(data_b, area, args.metric, args.rdm_type))
            offset_a += na
            offset_b += nb
            boundaries_a.append(offset_a)
            boundaries_b.append(offset_b)
        boundaries_a = boundaries_a[:-1]
        boundaries_b = boundaries_b[:-1]

        n_a_total = len(all_labels_a)
        n_b_total = len(all_labels_b)

        fig, ax = plt.subplots(figsize=(max(10, n_b_total * 0.35),
                                        max(8, n_a_total * 0.35)))
        im = ax.imshow(full_cross, aspect="auto", cmap="RdYlBu_r",
                       interpolation="nearest")

        ax.set_xticks(range(n_b_total))
        ax.set_xticklabels(all_labels_b, rotation=90, fontsize=5)
        ax.set_yticks(range(n_a_total))
        ax.set_yticklabels(all_labels_a, fontsize=5)

        ax.set_xlabel(f"Model B ({label_b})")
        ax.set_ylabel(f"Model A ({label_a})")

        for b in boundaries_a:
            ax.axhline(b - 0.5, color="white", linewidth=1, linestyle="--")
        for b in boundaries_b:
            ax.axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Pearson r")
        ax.set_title(
            f"Full Cross-Model RDM Similarity\n"
            f"A: {label_a}  vs  B: {label_b}\n"
            f"({args.metric}, {args.rdm_type})",
            fontsize=11,
        )
        plt.tight_layout()
        full_path = path.join(args.save_dir, "full_cross_model_similarity.png")
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {full_path}")

    print(f"\nDone. All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
