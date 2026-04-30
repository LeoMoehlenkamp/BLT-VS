"""
Extract features from a pretrained torchvision ResNet and compute RDMs.

Uses the same THINGS Drift stimulus set as the BLT-VS pipeline.
For each ResNet layer, features are globally average-pooled and an RDM is
computed (cosine distance, optionally ranked).

Output: .npz compatible with second_order_rdms_ann_vs_resnet.py

Usage:
  python extract_resnet_features_and_rdms.py \
      --csv_path <stimulus_information.csv> \
      --image_root <THINGS_drift/stimuli> \
      [--resnet_variant resnet50] \
      [--batch_size 32] \
      [--metric cosine] \
      [--save_dir analysis_outputs/resnet_rdms]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from tqdm import tqdm

# ============================================================
# ResNet layers we extract from (roughly hierarchical)
# ============================================================
RESNET_LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]

LAYER_DISPLAY = {
    "conv1":   "Conv1",
    "layer1":  "Layer1",
    "layer2":  "Layer2",
    "layer3":  "Layer3",
    "layer4":  "Layer4",
    "avgpool": "AvgPool",
}


# ============================================================
# Dataset — identical to extract_ann_features.py
# ============================================================
class StimuliDataset(Dataset):
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


# ============================================================
# Helpers
# ============================================================

def reorder_features_to_original_index(features, indices):
    reordered = np.empty_like(features)
    reordered[indices] = features
    return reordered


def compute_rdm(features, metric="cosine"):
    rdm_condensed = pdist(features, metric=metric)
    rdm_square = squareform(rdm_condensed)
    return rdm_condensed, rdm_square


def quick_sanity_check(rdm, name="RDM"):
    print(f"\nSanity check for {name}")
    print(f"  shape: {rdm.shape}")
    print(f"  min:   {np.min(rdm):.6f}")
    print(f"  max:   {np.max(rdm):.6f}")
    print(f"  mean:  {np.mean(rdm):.6f}")
    print(f"  NaN:   {np.isnan(rdm).any()}")
    print(f"  sym:   {np.allclose(rdm, rdm.T, atol=1e-6)}")
    print(f"  diag0: {np.allclose(np.diag(rdm), 0, atol=1e-6)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract pretrained ResNet features + compute RDMs"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to stimulus_information.csv")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory of stimulus images")
    parser.add_argument("--resnet_variant", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Which ResNet variant to use")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric for first-order RDMs")
    parser.add_argument("--save_dir", type=str,
                        default="analysis_outputs/resnet_rdms")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load pretrained ResNet
    # --------------------------------------------------------
    print(f"\nLoading pretrained {args.resnet_variant} ...")
    model_fn = getattr(models, args.resnet_variant)
    model = model_fn(weights="IMAGENET1K_V1")
    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Register hooks to capture intermediate activations
    # --------------------------------------------------------
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach()
        return hook_fn

    hook_handles = []
    hook_handles.append(model.conv1.register_forward_hook(make_hook("conv1")))
    hook_handles.append(model.layer1.register_forward_hook(make_hook("layer1")))
    hook_handles.append(model.layer2.register_forward_hook(make_hook("layer2")))
    hook_handles.append(model.layer3.register_forward_hook(make_hook("layer3")))
    hook_handles.append(model.layer4.register_forward_hook(make_hook("layer4")))
    hook_handles.append(model.avgpool.register_forward_hook(make_hook("avgpool")))

    # --------------------------------------------------------
    # ImageNet normalization (standard for torchvision models)
    # --------------------------------------------------------
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    dataset = StimuliDataset(args.csv_path, args.image_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(dataset)} images")

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------
    features = {layer: [] for layer in RESNET_LAYERS}
    all_indices = []

    print("\nExtracting features ...")
    with torch.no_grad():
        for imgs, indices in tqdm(loader, file=sys.stdout):
            imgs = imgs.to(device)
            all_indices.extend(indices.numpy())

            activations.clear()
            _ = model(imgs)

            for layer in RESNET_LAYERS:
                act = activations[layer]
                # Global average pooling for conv/residual layers
                if act.dim() == 4:
                    feat = act.mean(dim=[2, 3])
                else:
                    feat = act.squeeze()
                features[layer].append(feat.cpu().numpy())

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # Concatenate
    all_indices = np.array(all_indices)
    for layer in RESNET_LAYERS:
        features[layer] = np.concatenate(features[layer], axis=0)
        print(f"  {layer}: {features[layer].shape}")

    # --------------------------------------------------------
    # Compute RDMs per layer
    # --------------------------------------------------------
    print(f"\nComputing RDMs (metric={args.metric}) ...")
    save_dict = {
        "indices": all_indices.astype(np.int32),
        "distance_metric": np.array(args.metric),
        "resnet_variant": np.array(args.resnet_variant),
        "layers": np.array(RESNET_LAYERS),
    }

    for layer in RESNET_LAYERS:
        feat = features[layer]

        # Restore original stimulus ordering
        feat_ordered = reorder_features_to_original_index(feat, all_indices)

        # Compute RDM
        rdm_condensed, rdm_square = compute_rdm(feat_ordered, metric=args.metric)

        # Ranked version
        rdm_ranked = rankdata(rdm_condensed)
        rdm_ranked_square = squareform(rdm_ranked)

        quick_sanity_check(rdm_square, name=f"{layer}_{args.metric}_raw")

        # Save RDMs
        save_dict[f"{layer}_rdm_{args.metric}_raw"] = rdm_square.astype(np.float32)
        save_dict[f"{layer}_rdm_{args.metric}_ranked"] = rdm_ranked_square.astype(np.float32)

        # Save features too (for potential re-use)
        save_dict[f"{layer}_features"] = feat_ordered.astype(np.float32)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    save_path = os.path.join(args.save_dir, f"{args.resnet_variant}_rdms.npz")
    np.savez_compressed(save_path, **save_dict)
    print(f"\nSaved ResNet RDMs to: {save_path}")


if __name__ == "__main__":
    main()
