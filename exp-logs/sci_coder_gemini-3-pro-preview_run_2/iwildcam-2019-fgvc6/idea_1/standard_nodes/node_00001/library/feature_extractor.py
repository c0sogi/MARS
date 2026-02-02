import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from library.config import Config
from library.utils import set_seed


class FeatureModel(nn.Module):
    """
    A wrapper around ResNet50 to extract concatenated Global Average
    and Global Max Pooling features.
    """

    def __init__(self):
        super(FeatureModel, self).__init__()
        # Load pre-trained ResNet50 with default ImageNet weights
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights=weights)

        # Remove the final pooling (avgpool) and fully connected (fc) layers
        # The children list of ResNet50 ends with [..., layer4, avgpool, fc]
        # We slice up to layer4 (inclusive)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Freeze all parameters in the backbone
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Input: (Batch, 3, H, W)
        x = self.features(x)
        # Output: (Batch, 2048, H/32, W/32) -> (Batch, 2048, 7, 7) for 224x224 input

        # Global Average Pooling
        avg_pool = torch.mean(x, dim=(2, 3))  # (Batch, 2048)

        # Global Max Pooling
        max_pool = torch.amax(x, dim=(2, 3))  # (Batch, 2048)

        # Concatenate features
        out = torch.cat([avg_pool, max_pool], dim=1)  # (Batch, 4096)
        return out


def run_extraction(dataloader, model, device):
    """
    Helper function to iterate over a dataloader and extract features.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            targets = batch[1]

            # Forward pass
            features = model(inputs)

            # Move to CPU and store
            all_features.append(features.cpu().numpy())

            # Store targets/IDs
            if isinstance(targets, torch.Tensor):
                # For train/val sets, targets are Tensors
                all_labels.append(targets.numpy())
            else:
                # For test set, targets are tuples of ID strings
                all_labels.extend(targets)

    # Aggregate results
    features_np = np.concatenate(all_features, axis=0)

    # Handle label aggregation based on type (Array of ints vs List of strings)
    if len(all_labels) > 0 and isinstance(all_labels[0], (np.ndarray, np.generic)):
        labels_np = np.concatenate(all_labels, axis=0)
    else:
        labels_np = np.array(all_labels)

    return features_np, labels_np


def get_features(dataloader, mode, load_cached_data=True):
    """
    Main function to obtain features for a specific dataset split.
    Implements caching logic using .npy files.

    Args:
        dataloader (DataLoader): The data loader for the dataset.
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (features, targets) where features is a numpy array of shape (N, 4096)
               and targets is a numpy array of labels (int) or IDs (str).
    """
    set_seed(Config.SEED)

    # Define cache paths based on mode
    if mode == "train":
        feat_path = Config.TRAIN_FEATURES
        target_path = Config.TRAIN_TARGETS
    elif mode == "val":
        feat_path = Config.VAL_FEATURES
        target_path = Config.VAL_TARGETS
    elif mode == "test":
        feat_path = Config.TEST_FEATURES
        target_path = Config.TEST_IDS
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(feat_path) and os.path.exists(target_path):
            print(f"Loading cached features for '{mode}' from {Config.CACHE_DIR}...")
            try:
                features = np.load(feat_path)
                targets = np.load(target_path)
                return features, targets
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print(f"Cache not found for '{mode}'. Starting extraction...")
    else:
        print(f"Forcing re-computation for '{mode}'...")

    # Initialize model
    device = torch.device(Config.DEVICE)
    print(f"Initializing ResNet50 Feature Extractor on {device}...")
    model = FeatureModel().to(device)

    # Run extraction
    print(f"Extracting features for {len(dataloader.dataset)} samples...")
    features, targets = run_extraction(dataloader, model, device)

    # Save to cache
    print(f"Saving features to {feat_path}...")
    np.save(feat_path, features)
    np.save(target_path, targets)

    return features, targets
