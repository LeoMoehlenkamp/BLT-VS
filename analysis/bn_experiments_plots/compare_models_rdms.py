"""
Compare RDM plots per layer across multiple models.

Configure the MODELS list below, then run:
    python analysis/bn_experiments_plots/compare_models_rdms.py

Each entry points to the *_ann_rdms.npz file (or directory containing it).
The script creates one PNG per layer, showing all models side by side.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
## CONFIGURE HERE
# ============================================================

# Each entry: (path_to_rdm_npz_or_dir, display_name)
MODELS = [
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms.npz", "BN-none-BU"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNnone_BU_TD\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158_ann_rdms.npz", "BN-none-BU-TD"),
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNnone_BU_Skip\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523_ann_rdms.npz", "BN-none-BU-Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNnone_BU_TD_Skip\RDMs\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019_cosine_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019_ann_rdms.npz", "BN-none-BU-TD-Skip"),
    # Add more models here...
]

OUTPUT_DIR = r"C:\Users\moehl\Logs\temp\rdm_comparison"

AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

# Which timestep to show. Set to "best_last" to use the last available timestep,
# or an integer like 11 to pick a specific one.
TIMESTEP = "best_last"


def find_npz(path_str):
    """If path is a directory, find *_ann_rdms.npz inside it."""
    if os.path.isfile(path_str) and path_str.endswith(".npz"):
        return path_str
    if os.path.isdir(path_str):
        matches = glob.glob(os.path.join(path_str, "*_ann_rdms.npz"))
        if matches:
            return matches[0]
    print(f"WARNING: No RDM npz found at {path_str}, skipping.")
    return None


def get_timestep(data, area, preferred):
    """Find the right timestep for an area."""
    key_prefix = f"{area}_t"
    available = []
    for key in data.files:
        if key.startswith(key_prefix) and key.endswith("_rdm_cosine_ranked"):
            t_str = key.replace(key_prefix, "").replace("_rdm_cosine_ranked", "")
            available.append(int(t_str))
    available.sort()

    if not available:
        return None

    if preferred == "best_last":
        return available[-1]
    elif isinstance(preferred, int) and preferred in available:
        return preferred
    else:
        return available[-1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all model data
    model_data = []
    for path_str, name in MODELS:
        npz_path = find_npz(path_str)
        if npz_path is None:
            continue
        data = np.load(npz_path, allow_pickle=True)
        model_data.append((name, data))

    if not model_data:
        print("ERROR: No valid models found. Check your MODELS paths.")
        sys.exit(1)

    n_models = len(model_data)
    print(f"Loaded {n_models} models: {[m[0] for m in model_data]}")

    # Create one plot per area: rows = models, columns = timesteps
    for area in AREAS:
        # Collect all available timesteps across all models
        all_timesteps = set()
        for name, data in model_data:
            key_prefix = f"{area}_t"
            for key in data.files:
                if key.startswith(key_prefix) and key.endswith("_rdm_cosine_ranked"):
                    t_str = key.replace(key_prefix, "").replace("_rdm_cosine_ranked", "")
                    all_timesteps.add(int(t_str))

        all_timesteps = sorted(all_timesteps)
        if not all_timesteps:
            print(f"Skipping {area} — no timesteps found.")
            continue

        n_ts = len(all_timesteps)
        fig, axes = plt.subplots(n_models, n_ts, figsize=(2.5 * n_ts, 2.5 * n_models))

        # Ensure 2D array of axes
        if n_models == 1 and n_ts == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_ts == 1:
            axes = np.expand_dims(axes, axis=1)

        for row, (name, data) in enumerate(model_data):
            for col, t in enumerate(all_timesteps):
                ax = axes[row, col]
                key = f"{area}_t{t}_rdm_cosine_ranked"

                if key in data.files:
                    rdm = data[key]
                    ax.imshow(rdm, rasterized=True, interpolation="nearest")
                ax.axis("off")

                if row == 0:
                    ax.set_title(f"t{t}", fontsize=9)

            # Model name on the left — placed after tight_layout via fig.text

        fig.suptitle(f"{area} — RDM Comparison", fontsize=13)
        plt.tight_layout(rect=[0.12, 0, 1, 0.95])

        # Add model names after layout is finalized
        for row, (name, data) in enumerate(model_data):
            pos = axes[row, 0].get_position()
            y_center = (pos.y0 + pos.y1) / 2
            fig.text(0.10, y_center, name, va="center", ha="right", fontsize=9, fontweight="bold")

        save_path = os.path.join(OUTPUT_DIR, f"rdm_comparison_{area}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
