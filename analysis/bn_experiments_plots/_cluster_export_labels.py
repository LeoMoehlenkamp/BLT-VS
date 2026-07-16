"""Run ON THE CLUSTER. Reproduces the RDM row order and exports per-row labels.

Output: rdm_row_labels.csv with one row per RDM condition, in the SAME order
as the sorted first-order RDMs (V*_t*_rdm_* in the ann_rdms.npz).
"""
import os
import pickle
import numpy as np
import pandas as pd

CSV = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
PKL = "/share/klab/danthes/danthes/THINGS_Drift/results/rdm/monkeyF_mua_minithings/monkeyF-labels_filenames-sessions_0_1_2_3_4_5-rois_3-arrays_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16-baseline_0-standardize_1-metric_correlation-neural_mua.pkl"
OUT = os.path.expanduser("~/rdm_row_labels.csv")

SORT_COLS = [
    "animate", "body_parts", "human", "mammal", "non_mammal", "inanimate",
    "natural", "food", "fruit", "vegetable", "other_food", "plants",
    "other_natural", "artificial", "artificial_small", "tools",
    "artificial_small_other", "artificial_large", "furniture", "vehicles",
    "outside_large", "cat_id",
]

with open(PKL, "rb") as f:
    rdm_data = pickle.load(f)
label_col = rdm_data["data_cfg"]["labels"]
print("label_col =", label_col)

stim_info = pd.read_csv(CSV)
stim_info_sorted = stim_info.sort_values(SORT_COLS, ascending=False)
sel = stim_info_sorted[label_col].drop_duplicates()
indices = sel.index.values
allcols = stim_info.iloc[indices].copy()

# Keep the label column + the coarse taxonomy columns for grouping
keep = [label_col] + [c for c in SORT_COLS if c in allcols.columns]
allcols_out = allcols[keep].reset_index(drop=True)
allcols_out.to_csv(OUT, index=False)

print("n_rows =", len(allcols_out))
print("saved  =", OUT)
print(allcols_out.head(8).to_string())
