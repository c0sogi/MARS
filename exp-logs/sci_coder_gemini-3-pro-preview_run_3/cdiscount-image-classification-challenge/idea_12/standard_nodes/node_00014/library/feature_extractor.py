import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, IterableDataset

import library.config as config
from library.data_utils import BSONIterator
from library.hierarchy_utils import HierarchyMapper


# ==========================================
# MODEL DEFINITION
# ==========================================
class DualBackbone(nn.Module):
    """
    A dual-stream feature extractor using ResNet50 and EfficientNet-B0.
    Outputs a concatenated feature vector of size 2048 + 1280 = 3328.
    """

    def __init__(self):
        super(DualBackbone, self).__init__()

        # 1. ResNet50 (2048 dim)
        # We take all layers except the final FC layer.
        # resnet50.avgpool outputs (B, 2048, 1, 1)
        r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.r50_features = nn.Sequential(*list(r50.children())[:-1])

        # 2. EfficientNet-B0 (1280 dim)
        # efficientnet_b0.avgpool outputs (B, 1280, 1, 1)
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # The 'classifier' is the last module, we want everything before that.
        # efficientnet structure: features -> avgpool -> classifier
        self.eff_features = nn.Sequential(*list(eff.children())[:-1])

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 224, 224)

        # Stream 1: ResNet
        f1 = self.r50_features(x)  # (B, 2048, 1, 1)
        f1 = torch.flatten(f1, 1)  # (B, 2048)

        # Stream 2: EfficientNet
        f2 = self.eff_features(x)  # (B, 1280, 1, 1)
        f2 = torch.flatten(f2, 1)  # (B, 1280)

        # Concatenate
        out = torch.cat([f1, f2], dim=1)  # (B, 3328)
        return out


# ==========================================
# DATA LOADING UTILITIES
# ==========================================
class ShardedBSONDataset(IterableDataset):
    """
    Wraps BSONIterator to support multi-process data loading.
    Splits the metadata DataFrame among workers.
    """

    def __init__(self, bson_path, metadata_df):
        self.bson_path = bson_path
        self.metadata_df = metadata_df

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process execution
            iter_df = self.metadata_df
        else:
            # Multi-process execution: Shard the dataframe
            total_len = len(self.metadata_df)
            per_worker = int(np.ceil(total_len / float(worker_info.num_workers)))
            worker_id = worker_info.id

            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, total_len)

            if start_idx >= total_len:
                iter_df = pd.DataFrame()  # Empty for this worker
            else:
                iter_df = self.metadata_df.iloc[start_idx:end_idx]

        # Initialize the library's BSONIterator with the sharded dataframe
        iterator = BSONIterator(self.bson_path, iter_df)
        return iter(iterator)


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the batch of images for efficient GPU processing.

    Args:
        batch: List of tuples (_id, [img_tensors], category_id)

    Returns:
        ids: numpy array of product IDs
        imgs_tensor: Flattened tensor of all images (Total_Imgs, 3, H, W)
        counts: List of number of images per product (for un-pooling)
        cats: numpy array of category_ids (or None values)
    """
    batch_ids = []
    batch_imgs = []
    batch_counts = []
    batch_cats = []

    for _id, imgs, cat in batch:
        batch_ids.append(_id)
        batch_imgs.extend(imgs)
        batch_counts.append(len(imgs))
        batch_cats.append(cat)

    # Stack all images into a single large batch
    if batch_imgs:
        imgs_tensor = torch.stack(batch_imgs)
    else:
        imgs_tensor = torch.empty(0, 3, config.IMG_SIZE, config.IMG_SIZE)

    return np.array(batch_ids), imgs_tensor, batch_counts, np.array(batch_cats)


# ==========================================
# EXTRACTION LOGIC
# ==========================================
def extract_dataset(
    metadata_path, bson_path, model, mapper, is_test=False, desc="Dataset"
):
    """
    Processes a specific dataset (Train/Val/Test), extracts features, and returns arrays.
    """
    print(f"Preparing to extract features for: {desc}")

    # 1. Load Metadata
    df = pd.read_csv(metadata_path)

    # Debugging Subset
    if config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG MODE: Limiting {desc} to {config.DEBUG_SAMPLE_SIZE} samples.")
        df = df.iloc[: config.DEBUG_SAMPLE_SIZE]

    print(f"Processing {len(df)} records from {os.path.basename(bson_path)}...")

    # 2. Setup DataLoader
    dataset = ShardedBSONDataset(bson_path, df)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE_EXTRACT,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Storage
    all_features = []
    all_ids = []
    all_labels = []  # L3 indices

    model.eval()

    # 4. Inference Loop
    with torch.no_grad():
        for batch_idx, (ids, imgs, counts, cats) in enumerate(loader):
            imgs = imgs.to(config.DEVICE)

            # Forward pass (all images flattened)
            # Output: (N_total_images, 3328)
            features_flat = model(imgs)

            # Split back into products
            # torch.split takes a list of sizes
            features_split = torch.split(features_flat, counts)

            # Mean Pooling
            # Stack the means of each product's images
            # Handle case where a product might have 0 images (though unlikely in this dataset)
            pooled_features = []
            for f in features_split:
                if f.size(0) > 0:
                    pooled_features.append(f.mean(dim=0))
                else:
                    # Fallback for empty image list: zero vector
                    pooled_features.append(
                        torch.zeros(config.INPUT_DIM, device=config.DEVICE)
                    )

            pooled_batch = torch.stack(pooled_features).cpu().numpy()

            # Store
            all_features.append(pooled_batch)
            all_ids.append(ids)

            if not is_test:
                # Map raw category_id to L3 index
                # Vectorized mapping is faster, but we have a mix of pandas/numpy/dict
                # Doing list comprehension with the mapper
                l3_indices = [mapper.get_l3_index(c) for c in cats]
                all_labels.append(np.array(l3_indices, dtype=np.int32))

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed batch {batch_idx + 1}...")

    # 5. Concatenate
    final_features = np.concatenate(all_features, axis=0)
    final_ids = np.concatenate(all_ids, axis=0)

    print(f"Finished {desc}. Features shape: {final_features.shape}")

    if is_test:
        return final_features, final_ids, None
    else:
        final_labels = np.concatenate(all_labels, axis=0)
        return final_features, final_ids, final_labels


def extract_and_cache_features(load_cached_data=True):
    """
    Main entry point for feature extraction.
    Checks for cached .npy files. If missing or forced, runs extraction.
    """
    # Define file paths
    files_check = {
        "train": [config.TRAIN_FEATURES, config.TRAIN_IDS, config.TRAIN_LABELS_L3],
        "val": [config.VAL_FEATURES, config.VAL_IDS, config.VAL_LABELS_L3],
        "test": [config.TEST_FEATURES, config.TEST_IDS],
    }

    # Check cache existence
    cache_exists = True
    for group, paths in files_check.items():
        for p in paths:
            if not os.path.exists(p):
                cache_exists = False
                break

    if load_cached_data and cache_exists:
        print("All feature cache files found. Skipping extraction.")
        return

    print("Cache missing or reload requested. Starting feature extraction pipeline...")

    # Initialize Hierarchy Mapper
    mapper = HierarchyMapper(load_cached_data=True)

    # Initialize Model
    print("Initializing DualBackbone (ResNet50 + EfficientNetB0)...")
    model = DualBackbone().to(config.DEVICE)

    # ==========================
    # PROCESS TRAIN
    # ==========================
    feats, ids, labels = extract_dataset(
        config.TRAIN_META,
        config.TRAIN_BSON,
        model,
        mapper,
        is_test=False,
        desc="Training Set",
    )
    print(f"Saving training data to {config.WORKING_DIR}...")
    np.save(config.TRAIN_FEATURES, feats)
    np.save(config.TRAIN_IDS, ids)
    np.save(config.TRAIN_LABELS_L3, labels)

    # Clear memory
    del feats, ids, labels

    # ==========================
    # PROCESS VAL
    # ==========================
    feats, ids, labels = extract_dataset(
        config.VAL_META,
        config.TRAIN_BSON,
        model,
        mapper,
        is_test=False,
        desc="Validation Set",
    )
    print(f"Saving validation data to {config.WORKING_DIR}...")
    np.save(config.VAL_FEATURES, feats)
    np.save(config.VAL_IDS, ids)
    np.save(config.VAL_LABELS_L3, labels)

    del feats, ids, labels

    # ==========================
    # PROCESS TEST
    # ==========================
    feats, ids, _ = extract_dataset(
        config.TEST_META, config.TEST_BSON, model, mapper, is_test=True, desc="Test Set"
    )
    print(f"Saving test data to {config.WORKING_DIR}...")
    np.save(config.TEST_FEATURES, feats)
    np.save(config.TEST_IDS, ids)

    del feats, ids

    print("Feature extraction complete.")
