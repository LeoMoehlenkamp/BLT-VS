import os
from os import path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

# =========================
# LOCAL SETTINGS
# =========================
rdm_npz_path = r"C:\Users\moehl\Logs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_euclidean_ranked\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms.npz"
save_dir = r"C:\Users\moehl\Logs\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_euclidean_ranked"

# Welche first-order RDMs sollen aus dem npz genommen werden?
# z.B. "cosine" oder "euclidean"
distance_metric = "euclidean"

# z.B. "raw" oder "ranked"
rdm_type = "ranked"

# Plotgröße
single_figsize = (5, 4.5)
overview_scale = 4.0


def extract_matching_keys(npz_file, area, metric, rdm_type):
    pattern = re.compile(rf"^{area}_t(\d+)_rdm_{metric}_{rdm_type}$")
    matches = []

    for key in npz_file.files:
        m = pattern.match(key)
        if m:
            t = int(m.group(1))
            matches.append((t, key))

    matches.sort(key=lambda x: x[0])
    return matches


def rdm_to_vector(rdm):
    return squareform(rdm, checks=False)


def quick_check(mat, name):
    print(f"\n{name}")
    print("shape:", mat.shape)
    print("min:", np.min(mat))
    print("max:", np.max(mat))
    print("mean:", np.mean(mat))
    print("has NaN:", np.isnan(mat).any())
    print("symmetric:", np.allclose(mat, mat.T, atol=1e-6))
    print("diag ~ 0:", np.allclose(np.diag(mat), 0, atol=1e-6))


os.makedirs(save_dir, exist_ok=True)

data = np.load(rdm_npz_path, allow_pickle=True)
base_name = path.basename(rdm_npz_path).replace(".npz", "")
run_name = f"{base_name}__time_time_spearman__{distance_metric}_{rdm_type}"
out_dir = path.join(save_dir, run_name)
os.makedirs(out_dir, exist_ok=True)

overview_data = {}

for area in AREAS:
    matches = extract_matching_keys(data, area, distance_metric, rdm_type)

    if len(matches) == 0:
        print(f"No matching RDMs found for {area}")
        continue

    timesteps = [t for t, _ in matches]
    rdms = np.array([rdm_to_vector(data[key].astype(np.float64)) for _, key in matches])

    # genau wie beim Prof:
    # Zeilen = Timesteps, Spalten = RDM-Einträge
    spearman_rs, _ = spearmanr(rdms, axis=1)
    time_time_rdm = 1 - spearman_rs

    quick_check(time_time_rdm, f"{area} time-time RDM")

    # speichern npz
    save_npz = path.join(out_dir, f"{area}_time_time_spearman.npz")
    np.savez_compressed(
        save_npz,
        area=np.array(area),
        timesteps=np.array(timesteps, dtype=np.int32),
        spearman_r=np.array(spearman_rs, dtype=np.float32),
        time_time_rdm=np.array(time_time_rdm, dtype=np.float32),
        source_rdm_npz=np.array(rdm_npz_path),
        distance_metric=np.array(distance_metric),
        rdm_type=np.array(rdm_type),
    )

    # plot pro area
    plt.figure(figsize=single_figsize)
    plt.imshow(time_time_rdm, rasterized=True)

    tick_idx = np.arange(len(timesteps))
    tick_labels = [f"t{t}" for t in timesteps]

    plt.xticks(ticks=tick_idx, labels=tick_labels, rotation=90)
    plt.yticks(ticks=tick_idx, labels=tick_labels)
    plt.xlabel("timestep")
    plt.ylabel("timestep")
    plt.title(f"{area} time-time RDM")
    plt.colorbar(label="rank order distance (spearman)")
    plt.tight_layout()

    save_png = path.join(out_dir, f"{area}_time_time_spearman.png")
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close()

    overview_data[area] = {
        "rdm": time_time_rdm,
        "timesteps": timesteps
    }

# großes overview png
available_areas = [a for a in AREAS if a in overview_data]

if len(available_areas) > 0:
    n_cols = len(available_areas)
    fig, axes = plt.subplots(1, n_cols, figsize=(overview_scale * n_cols, overview_scale))

    if n_cols == 1:
        axes = [axes]

    for ax, area in zip(axes, available_areas):
        time_time_rdm = overview_data[area]["rdm"]
        timesteps = overview_data[area]["timesteps"]

        im = ax.imshow(time_time_rdm, rasterized=True)

        tick_idx = np.arange(len(timesteps))
        tick_labels = [f"t{t}" for t in timesteps]

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(area)
        ax.set_xlabel("timestep")
        ax.set_ylabel("timestep")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"time-time RDMs ({distance_metric}, {rdm_type})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    overview_path = path.join(out_dir, "all_areas_time_time_spearman.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"\nDone. Saved outputs to: {out_dir}")