import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


def _get_class_mapping(load_cached_data=True):
    """
    Generates or loads the class-to-index mapping based on the training set.
    Ensures consistent labeling across all splits.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_mapping.npy")

    if load_cached_data and os.path.exists(cache_path):
        classes = np.load(cache_path)
        return {cls: i for i, cls in enumerate(classes)}, classes

    # Load training metadata to determine classes
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_META_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    classes = sorted(df_train["species"].unique())

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, np.array(classes))

    return {cls: i for i, cls in enumerate(classes)}, classes


class LeafDataset(Dataset):
    """
    PyTorch Dataset for Leaf Images.
    Loads images and applies 4-view rotation augmentation (0, 90, 180, 270 degrees).
    """

    def __init__(
        self, dataframe, input_dir, class_to_idx=None, transform=None, is_test=False
    ):
        self.df = dataframe
        self.input_dir = input_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_test = is_test

        # Pre-compute paths and labels to avoid overhead in __getitem__
        self.file_paths = self.df["file_path"].values
        self.ids = self.df["id"].values

        if not self.is_test:
            self.species = self.df["species"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        img_id = self.ids[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # Images are binary (black leaf on white bg), but backbones expect RGB
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images (though metadata check passed)
            # Create a blank white image
            img = (
                np.ones((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8) * 255
            )
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(
            img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )

        # Generate 4 views
        views = []
        rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]

        for rot_code in rotations:
            if rot_code is None:
                view = img.copy()
            else:
                view = cv2.rotate(img, rot_code)

            # Apply transforms (ToTensor, Normalize)
            if self.transform:
                view = self.transform(view)
            else:
                view = torch.from_numpy(view.transpose(2, 0, 1)).float() / 255.0

            views.append(view)

        # Stack views: (4, C, H, W)
        images = torch.stack(views)

        # Get label
        if self.is_test:
            label = -1
        else:
            species_name = self.species[idx]
            label = self.class_to_idx[species_name]

        return images, label, img_id


def load_tabular_data(split, load_cached_data=True):
    """
    Loads tabular features (margin, shape, texture) and labels.
    Implements caching mechanism using .npy files.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached data if available.

    Returns:
        tuple: (features_array, labels_array, ids_array)
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_X = os.path.join(Config.CACHE_DIR, f"{split}_tabular_X.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"{split}_tabular_y.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"{split}_tabular_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_X) and os.path.exists(cache_ids):
            # For test split, y might not exist or be needed
            if split == "test" or os.path.exists(cache_y):
                X = np.load(cache_X)
                ids = np.load(cache_ids)
                y = np.load(cache_y) if split != "test" else None
                return X, y, ids

    # 2. Compute from scratch
    if split == "train":
        csv_path = Config.TRAIN_META_PATH
    elif split == "val":
        csv_path = Config.VAL_META_PATH
    elif split == "test":
        csv_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_csv(csv_path)

    # Extract features
    margin_cols = [c for c in df.columns if c.startswith("margin")]
    shape_cols = [c for c in df.columns if c.startswith("shape")]
    texture_cols = [c for c in df.columns if c.startswith("texture")]
    feature_cols = margin_cols + shape_cols + texture_cols

    X = df[feature_cols].values.astype(np.float32)
    ids = df["id"].values.astype(np.int64)

    y = None
    if split != "test":
        class_to_idx, _ = _get_class_mapping(load_cached_data)
        y = df["species"].map(class_to_idx).values.astype(np.int64)
        np.save(cache_y, y)

    # Save to cache
    np.save(cache_X, X)
    np.save(cache_ids, ids)

    return X, y, ids


def get_data_loaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    shuffle_train=True,
):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached class mapping.
        shuffle_train (bool): Whether to shuffle the training data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Standard ImageNet normalization
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get class mapping
    class_to_idx, _ = _get_class_mapping(load_cached_data)

    # Load DataFrames
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Create Datasets
    train_dataset = LeafDataset(
        df_train,
        Config.INPUT_DIR,
        class_to_idx=class_to_idx,
        transform=transform,
        is_test=False,
    )

    val_dataset = LeafDataset(
        df_val,
        Config.INPUT_DIR,
        class_to_idx=class_to_idx,
        transform=transform,
        is_test=False,
    )

    test_dataset = LeafDataset(
        df_test, Config.INPUT_DIR, class_to_idx=None, transform=transform, is_test=True
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
