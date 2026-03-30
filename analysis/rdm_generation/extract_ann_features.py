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

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import argparse

from blt_vs_model.training_code.models.helper_funcs import get_network_model

# ============================
# ARGUMENTS
# ============================

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--use_best", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()

MODEL_NAME = args.model_name
USE_BEST = bool(args.use_best)
BATCH_SIZE = args.batch_size

DEVICE = "cuda"

# ============================
# PATHS
# ============================

BASE_PATH = "/share/klab/danthes/lemoehlenkam/BLT-VS/logs"

CONFIG_DIR = os.path.join(BASE_PATH, "perf_logs", MODEL_NAME)
WEIGHTS_DIR = os.path.join(BASE_PATH, "net_params", MODEL_NAME)

CSV_PATH = "/share/klab/danthes/danthes/THINGS_Drift/datasets/stimulus_information.csv"
IMAGE_ROOT = "/share/klab/datasets/THINGS_drift/stimuli"

SAVE_PATH = f"analysis_outputs/ann_features/{MODEL_NAME}_features.npz"

AREAS = ["Retina","LGN","V1","V2","V3","V4","LOC"]

first_signal = {
    "Retina": 0,
    "LGN": 1,
    "V1": 2,
    "V2": 3,
    "V3": 4,
    "V4": 5,
    "LOC": 6
}

# ============================
# LOAD MODEL
# ============================

def load_model_from_name(model_name):

    config_path = os.path.join(CONFIG_DIR, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        hyp = json.load(f)

    print(f"Loaded config: {config_path}")

    model, _ = get_network_model(hyp)

    files = os.listdir(WEIGHTS_DIR)

    if USE_BEST:
        weight_files = [f for f in files if "BEST" in f]
    else:
        weight_files = [f for f in files if "LAST" in f]

    if len(weight_files) == 0:
        raise FileNotFoundError("No weights found")

    weight_path = os.path.join(WEIGHTS_DIR, weight_files[0])

    print(f"Loading weights: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")

    # Fix DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("Model ready.")

    return model, hyp

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
    T.Lambda(lambda x: 2*x - 1)  # 🔥 CRITICAL
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
# LOAD MODEL
# ============================

model, hyp = load_model_from_name(MODEL_NAME)

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

    for imgs, indices in tqdm(loader):

        imgs = imgs.to(DEVICE)

        # optional debug
        print(imgs.min().item(), imgs.max().item())

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

                feat = act.mean(dim=[2,3])

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

# Save indices for alignment
save_dict["indices"] = np.array(all_indices)

np.savez_compressed(SAVE_PATH, **save_dict)

print(f"Saved to: {SAVE_PATH}")

# ============================
# QUICK CHECK
# ============================

for key in save_dict:
    if key != "indices":
        print(f"{key}: {save_dict[key].shape}")
        break