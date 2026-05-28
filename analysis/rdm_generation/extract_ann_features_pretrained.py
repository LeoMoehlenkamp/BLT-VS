# ============================
# SETUP PATH
# ============================

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================
# IMPORTS
# ============================

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import argparse

from blt_vs_model import blt_vs_model

# ============================
# ARGUMENTS
# ============================

parser = argparse.ArgumentParser()
parser.add_argument("--training_dataset", type=str, default="ecoset",
                    choices=["ecoset", "imagenet"])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--output_name", type=str, default=None,
                    help="Override output model name. Defaults to blt_vs_pretrained_<dataset>")

args = parser.parse_args()

TRAINING_DATASET = args.training_dataset
BATCH_SIZE = args.batch_size
MODEL_NAME = args.output_name or f"blt_vs_pretrained_{TRAINING_DATASET}"

DEVICE = "cuda"

# ============================
# PATHS
# ============================

CSV_PATH = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
IMAGE_ROOT = "/share/klab/datasets/THINGS_drift/stimuli"

SAVE_PATH = f"analysis_outputs/ann_features/{MODEL_NAME}_features.npz"

AREAS = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]

# ============================
# FIRST SIGNAL (pretrained model has skip_connections=True, no bottlenecks)
# ============================

# With bio_unroll + skip V1->V4:
#   Retina:0, LGN:1, V1:2, V2:3, V3:4, V4:3, LOC:4
# Without bio_unroll (imagenet): all areas active from t=0
if TRAINING_DATASET == "ecoset":
    first_signal = {
        "Retina": 0, "LGN": 1, "V1": 2, "V2": 3,
        "V3": 4, "V4": 3, "LOC": 4,
    }
elif TRAINING_DATASET == "imagenet":
    first_signal = {
        "Retina": 0, "LGN": 0, "V1": 0, "V2": 0,
        "V3": 0, "V4": 0, "LOC": 0,
    }

print(f"First signal: {first_signal}")

# ============================
# DATASET
# ============================

class MonkeyStimuliDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

        self.image_paths = [
            os.path.join(image_root, fname.replace(".jpg", ".bmp"))
            for fname in self.df["filenames"]
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, idx

# ============================
# TRANSFORMS (MATCH TRAINING!)
# ============================

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Lambda(lambda x: 2*x - 1)
])

# ============================
# LOAD DATA
# ============================

dataset = MonkeyStimuliDataset(CSV_PATH, IMAGE_ROOT, transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

print(f"Loaded {len(dataset)} images")

# ============================
# LOAD PRETRAINED MODEL
# ============================

print(f"Loading pretrained BLT-VS ({TRAINING_DATASET}) from HuggingFace...")
model = blt_vs_model(pretrained=True, training_dataset=TRAINING_DATASET)
model = model.to(DEVICE)
model.eval()
print("Model ready.")

timesteps = list(range(model.timesteps))

# ============================
# FEATURE STORAGE
# ============================

features = {
    area: {t: [] for t in timesteps}
    for area in AREAS
}

all_indices = []

# ============================
# EXTRACTION
# ============================

with torch.no_grad():

    for imgs, indices in tqdm(loader, file=sys.stdout):

        imgs = imgs.to(DEVICE)

        all_indices.extend(indices.numpy())

        _, activations = model(
            imgs,
            extract_actvs=True,
            areas=AREAS,
            timesteps=timesteps,
            bu=True,
            td=False,
            concat=False
        )

        for area in activations:
            for t in activations[area]:

                if t < first_signal[area]:
                    continue

                act = activations[area][t]

                if act is None:
                    continue

                if isinstance(act, dict):
                    act = act["bu"]

                if torch.isnan(act).any():
                    raise ValueError(f"NaNs found in {area} at timestep {t}")

                feat = act.mean(dim=[2, 3])

                features[area][t].append(feat.detach().cpu())

# ============================
# CONCAT
# ============================

print("Concatenating features...")

for area in features:
    for t in features[area]:
        if len(features[area][t]) > 0:
            features[area][t] = torch.cat(features[area][t], dim=0).numpy()
        else:
            features[area][t] = None

# ============================
# CONSISTENCY CHECK
# ============================

n_expected = len(dataset)

for area in AREAS:
    for t in timesteps:
        arr = features[area][t]
        if arr is not None and arr.shape[0] != n_expected:
            raise ValueError(
                f"{area}_t{t} has {arr.shape[0]} samples, expected {n_expected}"
            )

# ============================
# SAVE
# ============================

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

save_dict = {}

for area in AREAS:
    for t in timesteps:
        key = f"{area}_t{t}"
        if features[area][t] is not None:
            save_dict[key] = features[area][t]

save_dict["indices"] = np.array(all_indices)

np.savez_compressed(SAVE_PATH, **save_dict)

print(f"Saved to: {SAVE_PATH}")
