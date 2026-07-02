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
    (r"C:\Users\moehl\Logs\Final\BU\BNnone_BU\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260316_210800_cosine_ranked_spearman\overview.png", "BNnone_BU"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV1V2_BU\BNV1V2_BU_12\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-V1V2-12__20260321_053846_cosine_ranked_spearman\overview.png", "BNV1V2_BU_12"),
    (r"C:\Users\moehl\Logs\Final\BU\BNV2V3_BU\BNV2V3_BU_8\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-V2V3-8__20260329_132907_cosine_ranked_spearman\overview.png", "BNV2V3_BU_8"),
    (r"C:\Users\moehl\Logs\Final\BU\BNall_BU\bnall64_BU\2nd_order\cosine_ranked_spearman\overview.png", "BNall_BU_64"),
]

PANELS = [
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNnone_BU_Skip\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-none__20260414_204523_cosine_ranked_spearman\overview.png", "BNnone_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-Skip\BNall_BU_Skip\BNall64_BU_Skip\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-bnall64skip__20260416_130242_cosine_ranked_spearman\overview.png", "BNall64_BU_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNnone_BU_TD\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD__20260421_120158_cosine_ranked_spearman\overview.png", "BNnone_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD\BNall_BU_TD\BNall64_BU_TD\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-TD__20260422_112005_cosine_ranked_spearman\overview.png", "BNall64_BU_ TD"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNnone_BU_TD_Skip\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bn-none_BU-TD-Skip__20260423_090019_cosine_ranked_spearman\overview.png", "BNnone_BU_ TD_Skip"),
    (r"C:\Users\moehl\Logs\Final\BU-TD-Skip\BNall_BU_TD_Skip\BNall32_BU_TD_Skip\2nd_order\blt_vs_bottleneck__miniecoset__ts12__bnall32_BU-TD-Skip__20260602_005408_cosine_ranked_spearman\overview.png", "BNall32_BU_ TD_Skip"),
]

PANELS = [
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143\2nd_order\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260525_153143_cosine_ranked_spearman\overview.png", "BNnone_BU_ TD_Skip_Ecoset"),
    (r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731\2nd_order\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260615_185731_cosine_ranked_spearman\overview.png", "BNall32_BU_TD_Skip_Ecoset"),
]

SAVE_PATH = r"C:\Users\moehl\Logs\Plots_BA\rdm_combined_Ecoset.png"

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

# Compute height ratios from actual image aspect ratios
ratios = [img.shape[0] / img.shape[1] for img, _ in images]
fig, axes = plt.subplots(n, 1, figsize=(12, sum(r * 12 for r in ratios)),
                         gridspec_kw={"hspace": 0.30})

if n == 1:
    axes = [axes]

for ax, (img, title) in zip(axes, images):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
    ax.axis("off")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"Saved: {SAVE_PATH}")
plt.close()
