import os
import torch
import torch.nn as nn
import numpy as np
from library.config import MODELS, DEVICE, CACHE_DIR
from library.utils import save_embeddings, load_embeddings
from library.data import create_dataloaders


def load_backbone(model_name):
    """
    Loads the specified model backbone, loads weights, and removes the classification head.

    Args:
        model_name (str): Key from library.config.MODELS.

    Returns:
        torch.nn.Module: The feature extractor model.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}")

    config = MODELS[model_name]
    constructor = config["constructor"]
    weights = config["weights"]

    print(f"Loading backbone: {model_name} with weights {weights}")
    model = constructor(weights=weights)

    # Remove classification head based on architecture type
    if "convnext" in model_name:
        # ConvNeXt classifier is a Sequential(LayerNorm2d, Flatten, Linear)
        # We want to keep LayerNorm and Flatten, replace Linear with Identity
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            # Check if the last layer is Linear
            if isinstance(model.classifier[-1], nn.Linear):
                model.classifier[-1] = nn.Identity()
            else:
                print(
                    "Warning: Expected Linear layer at end of ConvNeXt classifier, but found otherwise."
                )
        else:
            raise AttributeError(
                "ConvNeXt model structure mismatch: 'classifier' attribute not found or not Sequential."
            )

    elif "swin" in model_name:
        # Swin Transformer head is a Linear layer
        if hasattr(model, "head"):
            model.head = nn.Identity()
        else:
            raise AttributeError(
                "Swin model structure mismatch: 'head' attribute not found."
            )

    else:
        # Fallback for generic torchvision models (usually 'fc' or 'classifier')
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()

    model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings_with_tta(model, dataloader, device):
    """
    Extracts embeddings using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The feature extractor.
        dataloader (DataLoader): Data source.
        device (str): Computing device.

    Returns:
        tuple: (embeddings, labels, ids)
            - embeddings: np.ndarray of shape (N, D)
            - labels: np.ndarray of shape (N,) or None
            - ids: np.ndarray of shape (N,)
    """
    all_embeddings = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for images, labels, img_ids in dataloader:
            images = images.to(device)

            # 1. Forward pass original images
            features_orig = model(images)

            # 2. Forward pass flipped images (TTA)
            # Flip along width (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            features_flip = model(images_flipped)

            # 3. Average features
            features_avg = (features_orig + features_flip) / 2.0

            # Store results
            all_embeddings.append(features_avg.cpu().numpy())

            # Handle labels (might be dummy -1 for test)
            if isinstance(labels, torch.Tensor):
                all_labels.append(labels.numpy())
            else:
                # If labels is a list or other iterable
                all_labels.append(np.array(labels))

            # Handle IDs (usually a tuple or list of strings from DataLoader)
            all_ids.extend(img_ids)

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ids = np.array(all_ids)

    return embeddings, labels, ids


def run_feature_extraction(model_name, load_cached_data=True):
    """
    Orchestrates the feature extraction process with caching.

    Args:
        model_name (str): The model key.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            - 'train': (embeddings, labels, ids)
            - 'val': (embeddings, labels, ids)
            - 'test': (embeddings, labels, ids)
    """
    cache_prefix = MODELS[model_name]["cache_prefix"]

    # Define cache keys
    sets = ["train", "val", "test"]
    data = {}

    # Check if all cache files exist
    cache_hits = {}
    if load_cached_data:
        print(f"Checking cache for {model_name}...")
        for s in sets:
            prefix = f"{cache_prefix}_{s}"
            loaded = load_embeddings(prefix, CACHE_DIR)
            if loaded is not None:
                data[s] = loaded
                cache_hits[s] = True
            else:
                cache_hits[s] = False

    # If any set is missing from cache, we must load the model and extract
    missing_sets = [s for s in sets if not cache_hits.get(s, False)]

    if not missing_sets:
        print(f"All embeddings for {model_name} loaded from cache.")
        return data

    # Initialize Model and Dataloaders only if needed
    print(
        f"Cache miss for {missing_sets}. Starting feature extraction for {model_name}..."
    )

    model = load_backbone(model_name)
    train_loader, val_loader, test_loader, _ = create_dataloaders(model_name)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Process missing sets
    for s in missing_sets:
        print(f"Extracting features for {s} set...")
        loader = loaders[s]

        embeddings, labels, ids = extract_embeddings_with_tta(model, loader, DEVICE)

        # For test set, labels are dummy, but we save them anyway to maintain signature consistency
        # Save to cache
        prefix = f"{cache_prefix}_{s}"
        save_embeddings(embeddings, labels, ids, prefix, CACHE_DIR)

        # Store in return dict
        data[s] = (embeddings, labels, ids)

    print(f"Feature extraction for {model_name} completed.")
    return data
