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

# NEU: Parent Ordner mit 4 Runs
parent_dir = r"C:\Users\moehl\Logs\RDM_tests\bnall16__20260402_123451"
save_root = r"C:\Users\moehl\Logs\RDM_tests\bnall16__20260402_123451"

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


# alle Unterordner durchgehen
for run_folder in os.listdir(parent_dir):

    full_run_path = path.join(parent_dir, run_folder)

    if not path.isdir(full_run_path):
        continue

    print(f"\n==============================")
    print(f"Processing: {run_folder}")
    print(f"==============================")

    # metric + rdm_type automatisch erkennen
    if "cosine" in run_folder:
        metric = "cosine"
    elif "euclidean" in run_folder:
        metric = "euclidean"
    else:
        print("Skipping (no metric found)")
        continue

    if "ranked" in run_folder:
        rdm_type = "ranked"
    elif "raw" in run_folder:
        rdm_type = "raw"
    else:
        print("Skipping (no rdm_type found)")
        continue

    # 🔥 NPZ automatisch finden
    npz_files = [f for f in os.listdir(full_run_path) if f.endswith(".npz")]

    if len(npz_files) == 0:
        print("No npz found → skipping")
        continue

    rdm_npz_path = path.join(full_run_path, npz_files[0])
    print(f"Using NPZ: {rdm_npz_path}")

    data = np.load(rdm_npz_path, allow_pickle=True)

    run_name = f"{run_folder}_spearman"
    out_dir = path.join(save_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    overview_data = {}

    for area in AREAS:
        matches = extract_matching_keys(data, area, metric, rdm_type)

        if len(matches) == 0:
            print(f"No RDMs for {area}")
            continue

        timesteps = [t for t, _ in matches]
        rdms = np.array([rdm_to_vector(data[key].astype(np.float64)) for _, key in matches])

        spearman_rs, _ = spearmanr(rdms, axis=1)
        time_time_rdm = 1 - spearman_rs

        quick_check(time_time_rdm, f"{area}")

        # speichern
        np.savez_compressed(
            path.join(out_dir, f"{area}.npz"),
            time_time_rdm=time_time_rdm,
            timesteps=np.array(timesteps)
        )

        # plot
        plt.figure(figsize=single_figsize)
        plt.imshow(time_time_rdm)

        ticks = np.arange(len(timesteps))
        labels = [f"t{t}" for t in timesteps]

        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)
        plt.title(area)
        plt.colorbar()

        plt.savefig(path.join(out_dir, f"{area}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        overview_data[area] = (time_time_rdm, timesteps)

    # overview plot
    if len(overview_data) > 0:
        n = len(overview_data)
        fig, axes = plt.subplots(1, n, figsize=(overview_scale * n, overview_scale))

        if n == 1:
            axes = [axes]

        for ax, (area, (rdm, timesteps)) in zip(axes, overview_data.items()):
            im = ax.imshow(rdm)

            ticks = np.arange(len(timesteps))
            labels = [f"t{t}" for t in timesteps]

            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(area)

            fig.colorbar(im, ax=ax)

        plt.suptitle(f"{metric} - {rdm_type}")
        plt.tight_layout()

        plt.savefig(path.join(out_dir, "overview.png"), dpi=300, bbox_inches="tight")
        plt.close()

print("\n DONE ALL RUNS")