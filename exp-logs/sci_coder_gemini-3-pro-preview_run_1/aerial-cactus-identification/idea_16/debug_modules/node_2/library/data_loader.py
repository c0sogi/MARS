import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_file_sizes, normalize_file_sizes


def load_and_cache_images(df, cache_name, load_cached_data=True):
    """
    Loads images into RAM. Caches the resulting numpy array to disk to speed up future runs.

    Args:
        df (pd.DataFrame): Dataframe containing 'file_path' column.
        cache_name (str): Name of the cache file (e.g., 'train_imgs').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of images with shape (N, H, W, C) and dtype uint8.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            imgs = np.load(cache_path)
            # print(f"Loaded {cache_name} from cache. Shape: {imgs.shape}")
            return imgs
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Load from disk
    # print(f"Loading {len(df)} images from disk for {cache_name}...")
    img_list = []
    file_paths = df["file_path"].values

    for rel_path in file_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing images (should not happen with valid metadata)
            # Create a black image
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)

    imgs = np.array(img_list, dtype=np.uint8)

    # 3. Save to cache
    np.save(cache_path, imgs)
    # print(f"Saved {cache_name} to {cache_path}")

    return imgs


class CactusDataset(Dataset):
    def __init__(
        self, images, labels=None, film_feats=None, aux_targets=None, transform=None
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C), uint8.
            labels (np.ndarray, optional): Array of binary labels (N,).
            film_feats (np.ndarray, optional): Array of normalized file sizes for FiLM (N,).
            aux_targets (np.ndarray, optional): Array of normalized file sizes for Aux Loss (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.film_feats = film_feats
        self.aux_targets = aux_targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (should at least normalize/to_tensor)
            # But we assume transform is always provided in get_dataloaders
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare return values
        # We return a tuple that the training loop can unpack
        # Structure: image, label, film_feat, aux_target

        # Default values for inference/test mode if labels/aux not present
        label = torch.tensor(0.0, dtype=torch.float32)
        film_feat = torch.tensor(0.0, dtype=torch.float32)
        aux_target = torch.tensor(0.0, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.film_feats is not None:
            film_feat = torch.tensor(self.film_feats[idx], dtype=torch.float32)

        if self.aux_targets is not None:
            aux_target = torch.tensor(self.aux_targets[idx], dtype=torch.float32)

        return image, label, film_feat, aux_target


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 2. Load and Cache Images
    train_imgs = load_and_cache_images(train_df, "train_imgs", load_cached_data)
    val_imgs = load_and_cache_images(val_df, "val_imgs", load_cached_data)
    test_imgs = load_and_cache_images(test_df, "test_imgs", load_cached_data)

    # 3. Process File Sizes (for FiLM and MTL)
    # Get raw sizes
    train_sizes = get_file_sizes(
        train_df,
        root_dir=Config.INPUT_DIR,
        cache_name="train_fsizes",
        load_cached_data=load_cached_data,
    )
    val_sizes = get_file_sizes(
        val_df,
        root_dir=Config.INPUT_DIR,
        cache_name="val_fsizes",
        load_cached_data=load_cached_data,
    )
    test_sizes = get_file_sizes(
        test_df,
        root_dir=Config.INPUT_DIR,
        cache_name="test_fsizes",
        load_cached_data=load_cached_data,
    )

    # Normalize
    fs_data = normalize_file_sizes(train_sizes, val_sizes, test_sizes)

    # 4. Define Transforms
    # Train: Geometric Augmentations + Normalization
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Val/Test: Only Normalization
    eval_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 5. Create Datasets
    train_dataset = CactusDataset(
        images=train_imgs,
        labels=train_df["has_cactus"].values.astype(np.float32),
        film_feats=fs_data["train_film"],
        aux_targets=fs_data["train_aux"],
        transform=train_transform,
    )

    val_dataset = CactusDataset(
        images=val_imgs,
        labels=val_df["has_cactus"].values.astype(np.float32),
        film_feats=fs_data["val_film"],
        aux_targets=fs_data["val_aux"],
        transform=eval_transform,
    )

    test_dataset = CactusDataset(
        images=test_imgs,
        labels=None,  # No labels for test
        film_feats=fs_data["test_film"],
        aux_targets=None,  # No aux targets for test
        transform=eval_transform,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
