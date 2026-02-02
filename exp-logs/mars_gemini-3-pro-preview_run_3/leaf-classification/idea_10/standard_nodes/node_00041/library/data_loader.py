import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library import config


class LeafDataset(Dataset):
    """
    PyTorch Dataset for Leaf Images.
    Loads images, converts to RGB, resizes, and generates 4 canonical rotated views
    to enforce strict rotation invariance.

    Returns:
        images: Tensor of shape (4, 3, H, W) containing 0, 90, 180, 270 degree rotations.
        tabular: Tensor of shape (192,) containing margin, shape, and texture features.
        label: LongTensor (scalar) class index, or None if test set.
        id: Int image identifier.
    """

    def __init__(self, paths, tabular, labels=None, ids=None):
        self.paths = paths
        self.tabular = tabular
        self.labels = labels
        self.ids = ids

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        path = self.paths[idx]

        # Read as grayscale first (images are binary black/white)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for safety, though metadata verification ensures files exist
            img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)

        # Resize to target size
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        # Convert to RGB (3 channels) by replicating the single channel
        # This is necessary for pretrained models like DINOv2 and ConvNeXt
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to Tensor (C, H, W) in range [0, 1]
        img_tensor = transforms.functional.to_tensor(img)

        # Apply normalization
        img_tensor = self.normalize(img_tensor)

        # Generate 4 canonical views (0, 90, 180, 270 degrees)
        # torch.rot90 rotates in the spatial plane (dims 1 and 2)
        views = [
            img_tensor,  # 0 degrees
            torch.rot90(img_tensor, 1, [1, 2]),  # 90 degrees
            torch.rot90(img_tensor, 2, [1, 2]),  # 180 degrees
            torch.rot90(img_tensor, 3, [1, 2]),  # 270 degrees
        ]

        # Stack views -> (4, 3, 224, 224)
        images = torch.stack(views)

        # Prepare tabular features
        tab_feat = torch.tensor(self.tabular[idx], dtype=torch.float32)

        # Prepare ID
        id_val = self.ids[idx]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return images, tab_feat, label, id_val
        else:
            return images, tab_feat, id_val


def load_data(load_cached_data=True):
    """
    Loads dataset metadata, features, and labels.
    Implements caching using .npy files to avoid re-parsing CSVs and re-encoding labels.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing numpy arrays for paths, tabular features, labels, and ids
              for train, val, and test splits. Also includes 'classes' array.
    """
    # Define cache file paths
    cache_files = {
        "train_paths": os.path.join(config.CACHE_DIR, "train_paths.npy"),
        "train_tabular": os.path.join(config.CACHE_DIR, "train_tabular.npy"),
        "train_labels": os.path.join(config.CACHE_DIR, "train_labels.npy"),
        "train_ids": os.path.join(config.CACHE_DIR, "train_ids.npy"),
        "val_paths": os.path.join(config.CACHE_DIR, "val_paths.npy"),
        "val_tabular": os.path.join(config.CACHE_DIR, "val_tabular.npy"),
        "val_labels": os.path.join(config.CACHE_DIR, "val_labels.npy"),
        "val_ids": os.path.join(config.CACHE_DIR, "val_ids.npy"),
        "test_paths": os.path.join(config.CACHE_DIR, "test_paths.npy"),
        "test_tabular": os.path.join(config.CACHE_DIR, "test_tabular.npy"),
        "test_ids": os.path.join(config.CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(config.CACHE_DIR, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading data from cache at {config.CACHE_DIR}...")
        data = {}
        for k, v in cache_files.items():
            # allow_pickle=True is required for loading object arrays (strings)
            data[k] = np.load(v, allow_pickle=True)
        return data

    print("Cache not found or disabled. Processing metadata from CSVs...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Load Metadata CSVs
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Identify tabular feature columns (margin, shape, texture)
    feature_cols = [
        c for c in df_train.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Generate Class Mapping from Training Data
    classes = sorted(df_train["species"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Helper function to process each dataframe
    def process_df(df, is_test=False):
        # Generate full file paths
        # metadata contains relative path "images/{id}.jpg", config.INPUT_DIR is "input"
        paths = (
            df["file_path"].apply(lambda x: os.path.join(config.INPUT_DIR, x)).values
        )

        # Extract tabular features
        tabular = df[feature_cols].values.astype(np.float32)

        # Extract IDs
        ids = df["id"].values

        labels = None
        if not is_test:
            labels = df["species"].map(class_to_idx).values.astype(np.int64)

        return paths, tabular, labels, ids

    # Process splits
    train_paths, train_tab, train_y, train_ids = process_df(df_train)
    val_paths, val_tab, val_y, val_ids = process_df(df_val)
    test_paths, test_tab, _, test_ids = process_df(df_test, is_test=True)

    # Save processed arrays to cache
    np.save(cache_files["train_paths"], train_paths)
    np.save(cache_files["train_tabular"], train_tab)
    np.save(cache_files["train_labels"], train_y)
    np.save(cache_files["train_ids"], train_ids)

    np.save(cache_files["val_paths"], val_paths)
    np.save(cache_files["val_tabular"], val_tab)
    np.save(cache_files["val_labels"], val_y)
    np.save(cache_files["val_ids"], val_ids)

    np.save(cache_files["test_paths"], test_paths)
    np.save(cache_files["test_tabular"], test_tab)
    np.save(cache_files["test_ids"], test_ids)

    np.save(cache_files["classes"], np.array(classes))

    print(f"Data processed and saved to {config.CACHE_DIR}")

    return {
        "train_paths": train_paths,
        "train_tabular": train_tab,
        "train_labels": train_y,
        "train_ids": train_ids,
        "val_paths": val_paths,
        "val_tabular": val_tab,
        "val_labels": val_y,
        "val_ids": val_ids,
        "test_paths": test_paths,
        "test_tabular": test_tab,
        "test_ids": test_ids,
        "classes": np.array(classes),
    }
