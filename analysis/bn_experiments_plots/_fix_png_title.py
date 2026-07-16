"""One-off: overwrite the (mislabeled) title on a rendered PNG.

Covers the top title strip with white and redraws the corrected title,
leaving the rest of the plot untouched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

PNG = r"C:\Users\moehl\Logs\Final\Ecoset\blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731\Monkey RDMs\cosine_ranked\blt_vs_bottleneck__ecoset__ts12__bn-none_BU-TD-Skip__20260615_185731__cosine_ranked\summary_best_corr_per_area.png"

NEW_TITLE = "Best ANN match per area \u2013 blt_vs_bottleneck__ecoset__ts12__bnall32_BU-TD-Skip__20260615_185731"

STRIP_PX = 100      # height of the top strip to cover (pixels)
TITLE_Y_PX = 52     # vertical centre for the new title text (pixels)
FONTSIZE = 30       # points

img = mpimg.imread(PNG)
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
ax.text(w / 2, TITLE_Y_PX, NEW_TITLE, ha="center", va="center",
        fontsize=FONTSIZE, color="black", zorder=3)

fig.savefig(PNG, dpi=dpi)
print("Overwrote title in:", PNG)
plt.close()
