"""
Extract features from a pretrained torchvision ResNet and compute RDMs.

Uses the same THINGS Drift stimulus set as the BLT-VS pipeline.
Hooks are registered on every conv layer inside each residual block,
giving fine-grained hierarchical representations (e.g. layer1.0.conv1,
layer1.0.conv2, layer1.0.conv3, layer1.1.conv1, ...).

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
from sklearn.decomposition import PCA
from tqdm import tqdm


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
    parser.add_argument("--pca_components", type=int, default=1000,
                        help="Number of PCA components (0 = no PCA)")
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
    # Discover all sub-layers and register hooks
    #
    # Hook on conv layers to match the TIMM pkl format.
    # For BasicBlock (ResNet18/34): conv1, conv2
    # For Bottleneck (ResNet50+):   conv1, conv2, conv3
    # --------------------------------------------------------
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach()
        return hook_fn

    hook_handles = []
    layer_names = []

    # 1) conv1 (stem)
    hook_handles.append(model.conv1.register_forward_hook(make_hook("conv1")))
    layer_names.append("conv1")

    # 2) Inside each residual block — hook on conv layers
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model, stage_name)
        for block_idx, block in enumerate(stage):
            # Hook each conv layer in the block
            for conv_name in ["conv1", "conv2", "conv3"]:
                if hasattr(block, conv_name):
                    full_name = f"{stage_name}.{block_idx}.{conv_name}"
                    conv_module = getattr(block, conv_name)
                    hook_handles.append(conv_module.register_forward_hook(make_hook(full_name)))
                    layer_names.append(full_name)

    # 3) avgpool
    hook_handles.append(model.avgpool.register_forward_hook(make_hook("avgpool")))
    layer_names.append("avgpool")

    print(f"\nRegistered hooks on {len(layer_names)} layers:")
    for ln in layer_names:
        print(f"  {ln}")

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
    features = {layer: [] for layer in layer_names}
    all_indices = []

    print("\nExtracting features ...")
    with torch.no_grad():
        for imgs, indices in tqdm(loader, file=sys.stdout):
            imgs = imgs.to(device)
            all_indices.extend(indices.numpy())

            activations.clear()
            _ = model(imgs)

            for layer in layer_names:
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
    for layer in layer_names:
        features[layer] = np.concatenate(features[layer], axis=0)
        print(f"  {layer}: {features[layer].shape}")

    # --------------------------------------------------------
    # Compute RDMs per layer (with optional PCA)
    # --------------------------------------------------------
    print(f"\nComputing RDMs (metric={args.metric}, pca={args.pca_components}) ...")
    save_dict = {
        "indices": all_indices.astype(np.int32),
        "distance_metric": np.array(args.metric),
        "resnet_variant": np.array(args.resnet_variant),
        "layers": np.array(layer_names),
        "pca_components": np.array(args.pca_components),
    }

    for layer in layer_names:
        feat = features[layer]

        # Restore original stimulus ordering
        feat_ordered = reorder_features_to_original_index(feat, all_indices)

        # Apply PCA if requested and feature dim > n_components
        if args.pca_components > 0 and feat_ordered.shape[1] > args.pca_components:
            pca = PCA(n_components=args.pca_components)
            feat_ordered = pca.fit_transform(feat_ordered)
            print(f"  {layer}: PCA {feat.shape[1]} -> {feat_ordered.shape[1]}")
        else:
            print(f"  {layer}: no PCA (dim={feat_ordered.shape[1]})")

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
