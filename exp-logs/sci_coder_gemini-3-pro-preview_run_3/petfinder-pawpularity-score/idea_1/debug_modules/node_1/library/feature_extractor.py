import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from library.config import Config
from library.data_loader import get_dataloaders
from library.utils import set_seed


def extract_features_from_loader(loader, model, device):
    """
    Iterates over a dataloader to extract image features using the model,
    and collects metadata, targets, and IDs.

    Args:
        loader (DataLoader): The PyTorch DataLoader.
        model (nn.Module): The feature extractor model.
        device (torch.device): The computation device.

    Returns:
        tuple: (img_features, meta_features, targets, ids) as numpy arrays.
    """
    model.eval()

    img_features_list = []
    meta_features_list = []
    targets_list = []
    ids_list = []

    with torch.no_grad():
        for images, meta, targets, ids in loader:
            images = images.to(device)

            # Forward pass: features -> global_pool -> flatten -> identity
            # Output shape: (Batch_Size, 1280)
            outputs = model(images)

            img_features_list.append(outputs.cpu().numpy())
            meta_features_list.append(meta.numpy())
            targets_list.append(targets.numpy())
            ids_list.extend(ids)

    # Concatenate all batches
    if len(img_features_list) > 0:
        img_features = np.concatenate(img_features_list, axis=0)
        meta_features = np.concatenate(meta_features_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        ids = np.array(ids_list)
    else:
        # Handle empty loader case if necessary
        img_features = np.array([])
        meta_features = np.array([])
        targets = np.array([])
        ids = np.array([])

    return img_features, meta_features, targets, ids


def extract_features(load_cached_data=True):
    """
    Main function to extract features for Train, Val, and Test sets.
    Implements caching to avoid re-computation.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy cache.

    Returns:
        dict: A dictionary containing tuples for each split:
              'train': (X_img, X_meta, y)
              'val':   (X_img, X_meta, y)
              'test':  (X_img, X_meta, ids)
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths for easy checking
    cache_paths = [
        Config.CACHE_TRAIN_FEATURES,
        Config.CACHE_TRAIN_META,
        Config.CACHE_TRAIN_TARGETS,
        Config.CACHE_VAL_FEATURES,
        Config.CACHE_VAL_META,
        Config.CACHE_VAL_TARGETS,
        Config.CACHE_TEST_FEATURES,
        Config.CACHE_TEST_META,
        Config.CACHE_TEST_IDS,
    ]

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_paths)

    if load_cached_data and cache_exists:
        print("Loading extracted features from cache...")
        try:
            train_data = (
                np.load(Config.CACHE_TRAIN_FEATURES),
                np.load(Config.CACHE_TRAIN_META),
                np.load(Config.CACHE_TRAIN_TARGETS),
            )
            val_data = (
                np.load(Config.CACHE_VAL_FEATURES),
                np.load(Config.CACHE_VAL_META),
                np.load(Config.CACHE_VAL_TARGETS),
            )
            test_data = (
                np.load(Config.CACHE_TEST_FEATURES),
                np.load(Config.CACHE_TEST_META),
                np.load(Config.CACHE_TEST_IDS, allow_pickle=True),
            )
            return {"train": train_data, "val": val_data, "test": test_data}
        except Exception as e:
            print(f"Error loading cache: {e}. Re-computing features.")

    print("Starting feature extraction pipeline...")

    # 1. Setup Model
    device = torch.device(Config.DEVICE)
    # Load MobileNetV2 with default ImageNet weights
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Replace classifier with Identity to get the 1280-dim feature vector
    # MobileNetV2 structure: features -> adaptive_avg_pool2d -> flatten -> classifier
    model.classifier = nn.Identity()

    model.to(device)

    # 2. Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Extract Features
    print("Extracting Training features...")
    train_img, train_meta, train_y, _ = extract_features_from_loader(
        train_loader, model, device
    )

    print("Extracting Validation features...")
    val_img, val_meta, val_y, _ = extract_features_from_loader(
        val_loader, model, device
    )

    print("Extracting Test features...")
    test_img, test_meta, _, test_ids = extract_features_from_loader(
        test_loader, model, device
    )

    # 4. Save to Cache
    print("Saving features to cache...")
    np.save(Config.CACHE_TRAIN_FEATURES, train_img)
    np.save(Config.CACHE_TRAIN_META, train_meta)
    np.save(Config.CACHE_TRAIN_TARGETS, train_y)

    np.save(Config.CACHE_VAL_FEATURES, val_img)
    np.save(Config.CACHE_VAL_META, val_meta)
    np.save(Config.CACHE_VAL_TARGETS, val_y)

    np.save(Config.CACHE_TEST_FEATURES, test_img)
    np.save(Config.CACHE_TEST_META, test_meta)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return {
        "train": (train_img, train_meta, train_y),
        "val": (val_img, val_meta, val_y),
        "test": (test_img, test_meta, test_ids),
    }
