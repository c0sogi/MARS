import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    NUM_WORKERS,
)
from library.model import (
    ResNetBackbone,
    RawImageDataset,
    collate_raw_images,
    get_transforms,
)


def extract_dataset_features(
    metadata_path,
    bson_path,
    save_dir,
    split_name,
    batch_size=128,
    load_cached_data=True,
):
    """
    Extracts features from the dataset using ResNet50 and caches them as float16 .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        bson_path (str): Path to the source BSON file.
        save_dir (str): Directory where cache files will be stored.
        split_name (str): Identifier for the split (e.g., 'train', 'val', 'test').
        batch_size (int): Batch size for the feature extractor.
        load_cached_data (bool): If True, attempts to load from disk before computing.

    Returns:
        tuple: (features, indices, labels, ids)
            - features (np.ndarray): Flattened float16 feature array [Total_Images, 2048].
            - indices (np.ndarray): Index array [Num_Products, 2] containing (start_idx, count).
            - labels (np.ndarray or None): Category IDs [Num_Products]. None for test set.
            - ids (np.ndarray): Product IDs [Num_Products].
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define cache file paths
    feat_path = os.path.join(save_dir, f"{split_name}_features.npy")
    idx_path = os.path.join(save_dir, f"{split_name}_index.npy")
    label_path = os.path.join(save_dir, f"{split_name}_labels.npy")
    id_path = os.path.join(save_dir, f"{split_name}_ids.npy")

    # Check if cache exists (Features, Index, and IDs are mandatory)
    cache_valid = (
        os.path.exists(feat_path)
        and os.path.exists(idx_path)
        and os.path.exists(id_path)
    )

    if load_cached_data and cache_valid:
        print(f"Loading cached features for '{split_name}' from {save_dir}...")
        features = np.load(feat_path)
        indices = np.load(idx_path)
        ids = np.load(id_path)

        labels = None
        if os.path.exists(label_path):
            labels = np.load(label_path)

        return features, indices, labels, ids

    # --- Computation Path ---
    print(f"Extracting features for '{split_name}' from {bson_path}...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Initialize Model (Frozen ResNet50)
    model = ResNetBackbone().to(DEVICE)
    model.eval()

    # Initialize Dataset and Loader
    transforms = get_transforms()
    dataset = RawImageDataset(metadata_path, bson_path, transform=transforms)

    # Use collate_raw_images to handle variable number of images per product
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_raw_images,
        pin_memory=True,
    )

    all_features = []
    all_counts = []

    # Inference Loop
    with torch.no_grad():
        for imgs, counts, _ in loader:
            imgs = imgs.to(DEVICE)

            # Forward pass: [Sum_N, 3, 224, 224] -> [Sum_N, 2048]
            feats = model(imgs)

            # Cast to float16 to optimize memory usage (2048 float32 is heavy)
            feats_np = feats.cpu().numpy().astype(np.float16)

            all_features.append(feats_np)
            all_counts.extend(counts)

    # Concatenate all features into a single large array
    if all_features:
        flat_features = np.concatenate(all_features, axis=0)
    else:
        flat_features = np.empty((0, 2048), dtype=np.float16)

    # Build Index Array [Start, Count] for ragged access
    counts_arr = np.array(all_counts, dtype=np.int32)
    starts_arr = np.concatenate(([0], np.cumsum(counts_arr)[:-1])).astype(np.int32)
    indices = np.stack([starts_arr, counts_arr], axis=1)

    # Extract IDs and Labels from metadata
    ids = df["_id"].values.astype(np.int64)

    labels = None
    if "category_id" in df.columns:
        labels = df["category_id"].values.astype(np.int64)

    # Save to Cache
    print(f"Saving cached files to {save_dir}...")
    np.save(feat_path, flat_features)
    np.save(idx_path, indices)
    np.save(id_path, ids)

    if labels is not None:
        np.save(label_path, labels)

    return flat_features, indices, labels, ids
