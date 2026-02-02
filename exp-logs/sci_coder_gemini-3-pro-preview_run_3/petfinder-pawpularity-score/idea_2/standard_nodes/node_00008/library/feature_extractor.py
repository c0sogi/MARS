import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from library import config, utils


class DualBackboneExtractor(nn.Module):
    """
    A PyTorch module that extracts features using two frozen backbones:
    Swin Transformer and EfficientNetV2.
    """

    def __init__(self):
        super(DualBackboneExtractor, self).__init__()
        # Initialize Swin Transformer (Global features)
        self.swin = timm.create_model(
            config.BACKBONE_SWIN, pretrained=True, num_classes=0
        )

        # Initialize EfficientNetV2 (Local details)
        self.effnet = timm.create_model(
            config.BACKBONE_EFFNET, pretrained=True, num_classes=0
        )

        # Freeze all parameters
        for param in self.swin.parameters():
            param.requires_grad = False
        for param in self.effnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from both backbones
        # x shape: (Batch, 3, 224, 224)
        f1 = self.swin(x)  # Shape: (Batch, Embed_Dim_Swin)
        f2 = self.effnet(x)  # Shape: (Batch, Embed_Dim_EffNet)

        # Concatenate features
        return torch.cat([f1, f2], dim=1)


def _extract_loop(dataloader, model, device):
    """
    Internal helper to iterate over dataloader and extract features.
    """
    model.eval()
    features_list = []
    targets_list = []

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)

            # Get image embeddings with TTA (Cite solution_lesson_node_00007)
            # Pass 1: Original Images
            emb_orig = model(images)

            # Pass 2: Horizontally Flipped Images
            # Tensor shape is (B, C, H, W), so dim=3 is Width
            images_flip = torch.flip(images, dims=[3])
            emb_flip = model(images_flip)

            # Average the embeddings to reduce variance
            img_embeddings = (emb_orig + emb_flip) / 2.0

            # Concatenate image embeddings with metadata features
            # img_embeddings: (B, D_img), metadata: (B, D_meta)
            full_features = torch.cat([img_embeddings, metadata], dim=1)

            features_list.append(full_features.cpu().numpy())
            targets_list.append(targets.numpy())

    # Concatenate all batches
    final_features = np.concatenate(features_list, axis=0)
    final_targets = np.concatenate(targets_list, axis=0)

    return final_features, final_targets


def get_features(dataloader, mode="train", load_cached_data=True):
    """
    Main function to get features and targets/ids.
    Handles caching logic: loads from disk if available, else computes and saves.

    Args:
        dataloader: PyTorch DataLoader.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (features, targets) for train/val, (features, ids) for test.
    """
    # Determine file paths based on mode
    if mode == "train":
        feature_path = config.TRAIN_FEATURES_PATH
        target_path = config.TRAIN_TARGETS_PATH
    elif mode == "val":
        feature_path = config.VAL_FEATURES_PATH
        target_path = config.VAL_TARGETS_PATH
    elif mode == "test":
        feature_path = config.TEST_FEATURES_PATH
        ids_path = config.TEST_IDS_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Attempt to load from cache
    if load_cached_data:
        if mode == "test":
            if os.path.exists(feature_path) and os.path.exists(ids_path):
                print(f"[{mode}] Loading cached features and IDs...")
                features = utils.load_numpy_array(feature_path)
                ids = utils.load_numpy_array(ids_path)
                return features, ids
        else:
            if os.path.exists(feature_path) and os.path.exists(target_path):
                print(f"[{mode}] Loading cached features and targets...")
                features = utils.load_numpy_array(feature_path)
                targets = utils.load_numpy_array(target_path)
                return features, targets

    # If cache miss or force reload, compute features
    print(
        f"[{mode}] Cache not found or reload requested. Starting feature extraction..."
    )

    device = config.DEVICE
    model = DualBackboneExtractor().to(device)

    features, targets = _extract_loop(dataloader, model, device)

    # Save features
    utils.save_numpy_array(feature_path, features)

    if mode == "test":
        # For test set, we need IDs instead of targets
        # Load IDs directly from the dataset to ensure alignment with debug/subset logic
        ids = dataloader.dataset.df["Id"].values

        # Save IDs
        utils.save_numpy_array(ids_path, ids)

        print(
            f"[{mode}] Extraction complete. Saved features to {feature_path} and IDs to {ids_path}"
        )
        return features, ids
    else:
        # Save targets for train/val
        utils.save_numpy_array(target_path, targets)

        print(
            f"[{mode}] Extraction complete. Saved features to {feature_path} and targets to {target_path}"
        )
        return features, targets
