"""One-off: overwrite the (mislabeled) title on a rendered PNG.

Covers the top title strip with white and redraws the corrected title,
leaving the rest of the plot untouched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

JOBS = [
    (
        r"C:\Users\moehl\Logs\Plots_BA\3.7\summary_best_corr_per_areabnall64BUSkip.png",
        "Best ANN match per area \u2013 blt_vs_bottleneck__miniecoset__ts12__bnall64_BU-Skip__20260416_130242",
        r"C:\Users\moehl\Logs\Plots_BA\3.7\summary_best_corr_per_areabnall64BUSkip_fixed.png",
    ),
]

STRIP_PX = 88       # height of the top strip to cover (pixels)
TITLE_Y_PX = 40     # vertical centre for the new title text (pixels)
FONTSIZE = 30       # points


def fix_title(png, new_title, out_png):
    img = mpimg.imread(png)
    h, w = img.shape[:2]
    dpi = 100.0

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=[0, w, h, 0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # White-out the old title
    ax.add_patch(Rectangle((0, 0), w, STRIP_PX, color="white", zorder=2))

    # Redraw corrected title
    ax.text(w / 2, TITLE_Y_PX, new_title, ha="center", va="center",
            fontsize=FONTSIZE, color="black", zorder=3)

    fig.savefig(out_png, dpi=dpi)
    print("Saved corrected copy to:", out_png)
    plt.close()


for png, new_title, out_png in JOBS:
    fix_title(png, new_title, out_png)
