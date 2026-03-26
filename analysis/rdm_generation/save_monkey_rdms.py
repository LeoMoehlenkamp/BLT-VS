import os
from os import path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FULL_PANEL_SIZE = (24, 6)


def get_rdm_design_sort_indices(stimulus_csv, reduce_to_column="category", return_values=False):
    stim_info = pd.read_csv(stimulus_csv)

    stim_info_sorted = stim_info.sort_values(
        [
            "animate",
            "body_parts",
            "human",
            "mammal",
            "non_mammal",
            "inanimate",
            "natural",
            "food",
            "fruit",
            "vegetable",
            "other_food",
            "plants",
            "other_natural",
            "artificial",
            "artificial_small",
            "tools",
            "artificial_small_other",
            "artificial_large",
            "furniture",
            "vehicles",
            "outside_large",
            "cat_id",
        ],
        ascending=False,
    )

    stim_info_select = stim_info_sorted[reduce_to_column]
    stim_info_select = stim_info_select.drop_duplicates()

    indices = stim_info_select.index.values
    stim_info_select_allcols = stim_info.iloc[indices]
    sort_idx = rankdata(stim_info_select.index.values).astype(int) - 1

    if not return_values:
        return sort_idx
    return sort_idx, stim_info_select.values, stim_info_select_allcols


rdm_path = "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_mua_minithings/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua.pkl"
stimulus_csv = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
t_select = np.arange(0, 160, 10)

savedir = "analysis_outputs/monkey_rdms/rdm_trajectory_panels_mua"
os.makedirs(savedir, exist_ok=True)

with open(rdm_path, "rb") as f:
    rdm_data = pickle.load(f)

sort_idx = get_rdm_design_sort_indices(
    stimulus_csv,
    return_values=False,
    reduce_to_column=rdm_data["data_cfg"]["labels"]
)

time = rdm_data["time"]
rdms = rdm_data["rdms"]

n_panels = len(t_select)
full_width = FULL_PANEL_SIZE[0]
panel_size = full_width / n_panels

plt.figure(figsize=(full_width, panel_size))

for i in range(n_panels):
    t_panel = t_select[i]
    idx = np.where(time == t_panel)[0][0]

    rdm = rdms[idx]
    rdm = rankdata(rdm)
    rdm = squareform(rdm)
    rdm = rdm[sort_idx][:, sort_idx]

    plt.subplot(1, n_panels, i + 1)
    plt.imshow(rdm, rasterized=True)
    plt.gca().axis("off")
    plt.title(f"{t_panel}ms")

save_path = path.join(savedir, path.basename(rdm_path)[:-4] + ".svg")
plt.savefig(save_path, dpi=800, bbox_inches="tight")
plt.close()

"""# optional: also save raw data for later use
np.savez_compressed(
    path.join(savedir, path.basename(rdm_path)[:-4] + ".npz"),
    time=np.array(time),
    rdms=np.array(rdms, dtype=object),
    sort_idx=sort_idx,
)"""

print(f"Saved panel plot to: {save_path}")