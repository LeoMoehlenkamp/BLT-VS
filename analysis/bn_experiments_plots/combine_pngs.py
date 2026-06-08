"""
Combine multiple PNG images into a single stacked figure with custom titles.

Configure PANELS below: each entry is (path_to_png, title_text).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
# CONFIGURE HERE
# ============================================================

PANELS = [
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_ann_rdms__time_time_spearman__cosine_ranked\all_areas_time_time_spearman.png", "BNnone_BU"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV1V2_BU\BNV1V2_BU_192\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-192__20260317_131444_cosine_ranked_spearman\overview.png", "BNV1V2_BU_192"),
    (r"C:\Users\moehl\Logs\Plots_BA\rdm3.png", "BNV1V2_BU_32"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV1V2_BU\BNV1V2_BU_12\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846_ann_rdms__time_time_spearman__cosine_ranked\all_areas_time_time_spearman.png", "BNV1V2_BU_12"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\rdm_combined.png"

# ============================================================
# COMBINE
# ============================================================

images = []
for path, title in PANELS:
    if not os.path.exists(path):
        print(f"WARNING: Not found, skipping: {path}")
        continue
    images.append((mpimg.imread(path), title))

if not images:
    print("ERROR: No images loaded.")
    exit(1)

n = len(images)
fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))

if n == 1:
    axes = [axes]

for ax, (img, title) in zip(axes, images):
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {SAVE_PATH}")
plt.close()
