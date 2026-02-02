import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import Mixup

from library.config import CFG


def get_transforms(data="train", size=384):
    """
    Returns the Albumentations transformations for the specified data split.

    Args:
        data (str): 'train' or 'valid'.
        size (int): Image size for resizing.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Same as valid for test/inference
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CassavaDataset(Dataset):
    def __init__(self, df, file_root, transform=None, output_label=True, seed=42):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image_id and optionally label.
            file_root (str): Root directory containing the images.
            transform (A.Compose): Albumentations transforms.
            output_label (bool): Whether to return the label.
            seed (int): Base seed for deterministic behavior.
        """
        self.df = df
        self.file_root = file_root
        self.transform = transform
        self.output_label = output_label
        self.seed = seed

        # Pre-extract lists to avoid pandas overhead in __getitem__
        self.image_ids = self.df["image_id"].values
        if self.output_label:
            self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Deterministic seeding based on index
        # This ensures that for a specific image index, the augmentation is the same
        # if we were to access it multiple times in a deterministic loop,
        # or distinct across indices.
        # Note: In a standard training loop, we might want random augmentations per epoch.
        # However, the requirement specifically asks for seeding with image index
        # for deterministic geometric transformations.
        seed = self.seed + idx
        random.seed(seed)
        np.random.seed(seed)

        image_id = self.image_ids[idx]
        # Construct full path
        file_path = os.path.join(self.file_root, image_id)

        # Load image
        img = cv2.imread(file_path)
        if img is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            img = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.output_label:
            label = self.labels[idx]
            return img, torch.tensor(label).long()
        else:
            return img


def load_metadata(mode, load_cached_data=True):
    """
    Loads metadata from CSV or Parquet cache.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    cache_path = os.path.join(CFG.output_dir, f"{mode}_meta.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to load from source
            pass

    # 2. Load from source CSV
    if mode == "train":
        csv_path = CFG.train_csv
    elif mode == "val":
        csv_path = CFG.val_csv
    elif mode == "test":
        csv_path = CFG.test_csv
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache for {mode}: {e}")

    return df


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation, and initializes MixUp.

    Args:
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, mixup_fn)
    """
    # Load Metadata
    train_df = load_metadata("train", load_cached_data=load_cached_data)
    val_df = load_metadata("val", load_cached_data=load_cached_data)

    # Debug mode: subset data
    if CFG.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Create Datasets
    train_dataset = CassavaDataset(
        train_df,
        CFG.train_root,
        transform=get_transforms("train", CFG.img_size),
        output_label=True,
        seed=CFG.seed,
    )

    val_dataset = CassavaDataset(
        val_df,
        CFG.train_root,  # Validation images are also in train_images folder
        transform=get_transforms("valid", CFG.img_size),
        output_label=True,
        seed=CFG.seed,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize MixUp
    mixup_fn = Mixup(
        mixup_alpha=CFG.mixup_alpha,
        cutmix_alpha=CFG.cutmix_alpha,
        prob=CFG.mixup_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=CFG.label_smoothing,
        num_classes=CFG.num_classes,
    )

    return train_loader, val_loader, mixup_fn
