import os
import torch
import numpy as np
import pandas as pd
from torch_scatter import scatter_mean, scatter_max
from library.config import Config
from library.model import ImageEncoder
from library.data_loader import get_bson_loader
from library.utils import seed_everything


def extract_and_aggregate(
    metadata_path,
    save_path_features,
    save_path_labels=None,
    save_path_ids=None,
    load_cached_data=True,
    limit=None,
    batch_size=128,
    num_workers=12,
):
    """
    Extracts features from images using ResNet-50 and aggregates them per product
    using Dual-Statistic Pooling (Mean + Max).

    Args:
        metadata_path (str): Path to the dataset metadata CSV.
        save_path_features (str): Path to save/load the features .npy file.
        save_path_labels (str, optional): Path to save/load the labels .npy file.
        save_path_ids (str, optional): Path to save/load the ids .npy file.
        load_cached_data (bool): If True, tries to load from disk first.
        limit (int, optional): Limit number of records (for debugging).
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (features, labels, ids) as numpy arrays.
               labels and ids may be None if their paths were not provided/needed.
    """
    seed_everything(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(save_path_features), exist_ok=True)

    # 1. Attempt to Load Cached Data
    if load_cached_data:
        features_exist = os.path.exists(save_path_features)
        labels_exist = (save_path_labels is None) or os.path.exists(save_path_labels)
        ids_exist = (save_path_ids is None) or os.path.exists(save_path_ids)

        if features_exist and labels_exist and ids_exist:
            print(f"Loading cached features from {save_path_features}...")
            features = np.load(save_path_features)

            labels = None
            if save_path_labels:
                labels = np.load(save_path_labels)

            ids = None
            if save_path_ids:
                ids = np.load(save_path_ids)

            return features, labels, ids
        else:
            print("Cached files not found or incomplete. Starting extraction...")

    # 2. Setup Extraction Pipeline
    device = Config.DEVICE

    # Initialize Model
    print("Initializing ResNet-50 backbone...")
    model = ImageEncoder().to(device)
    model.eval()

    # Initialize DataLoader
    print(f"Initializing DataLoader for {metadata_path}...")
    loader = get_bson_loader(
        metadata_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=limit,
    )

    # 3. Pre-allocate Memory
    # We read the metadata to get the exact size to avoid memory re-allocation spikes
    meta_df = pd.read_csv(metadata_path)
    if limit:
        meta_df = meta_df.iloc[:limit]
    n_samples = len(meta_df)

    # Dual-Statistic Pooling: 2048 (Mean) + 2048 (Max) = 4096
    feature_dim = Config.INPUT_DIM

    print(f"Allocating memory for {n_samples} samples (Dim: {feature_dim})...")
    features_arr = np.zeros((n_samples, feature_dim), dtype=np.float32)

    labels_arr = None
    if save_path_labels:
        labels_arr = np.zeros(n_samples, dtype=np.int64)

    ids_arr = None
    if save_path_ids:
        ids_arr = np.zeros(n_samples, dtype=np.int64)

    # 4. Extraction Loop
    print("Starting feature extraction...")
    ptr = 0

    with torch.no_grad():
        for batch in loader:
            # Unpack batch
            # images: (Total_Images_In_Batch, 3, 224, 224)
            # counts: (Batch_Size,) -> Number of images per product
            imgs = batch["images"].to(device)
            counts = batch["counts"].to(device)
            batch_ids = batch["ids"].numpy()
            batch_labels = batch["labels"].numpy()

            current_batch_size = len(counts)

            # Extract features for all images
            # Output: (Total_Images_In_Batch, 2048)
            raw_feats = model(imgs)

            # Aggregate per product
            # Create an index tensor: [0, 0, 1, 2, 2, 2, ...] based on counts
            batch_indices = torch.repeat_interleave(
                torch.arange(current_batch_size, device=device), counts
            )

            # Scatter Mean: (Batch_Size, 2048)
            mean_pool = scatter_mean(
                raw_feats, batch_indices, dim=0, dim_size=current_batch_size
            )

            # Scatter Max: (Batch_Size, 2048)
            # scatter_max returns (values, indices)
            max_pool, _ = scatter_max(
                raw_feats, batch_indices, dim=0, dim_size=current_batch_size
            )

            # Concatenate: (Batch_Size, 4096)
            agg_feats = torch.cat([mean_pool, max_pool], dim=1)

            # Store in pre-allocated arrays
            end_ptr = ptr + current_batch_size
            features_arr[ptr:end_ptr] = agg_feats.cpu().numpy()

            if labels_arr is not None:
                labels_arr[ptr:end_ptr] = batch_labels

            if ids_arr is not None:
                ids_arr[ptr:end_ptr] = batch_ids

            ptr = end_ptr

    # 5. Save to Disk
    print("Saving extracted features to disk...")
    np.save(save_path_features, features_arr)

    if save_path_labels:
        np.save(save_path_labels, labels_arr)

    if save_path_ids:
        np.save(save_path_ids, ids_arr)

    print("Extraction complete.")
    return features_arr, labels_arr, ids_arr
