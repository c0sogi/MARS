import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG,
    DEBUG_SIZE,
)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Humpback Whale Identification.
    Handles loading images, caching them to disk as .npy files for speed,
    and applying augmentations.
    """

    def __init__(
        self,
        df,
        transforms=None,
        cache_name="cache.npy",
        load_cache=True,
        label_encoder=None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'Id' columns.
            transforms (albumentations.Compose): Augmentations to apply.
            cache_name (str): Filename for the cache file in WORKING_DIR.
            load_cache (bool): Whether to attempt loading from cache.
            label_encoder (dict): Dictionary mapping label strings to integers.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.label_encoder = label_encoder
        self.cache_path = os.path.join(WORKING_DIR, cache_name)
        self.load_cache = load_cache

        # Load images into memory (RAM is sufficient: ~1.5GB for full dataset)
        self.images = self._load_data()

    def _load_data(self):
        """
        Loads images from cache if available, otherwise reads from disk,
        resizes, and saves to cache.
        """
        # 1. Try to load from cache
        if self.load_cache and os.path.exists(self.cache_path):
            try:
                images = np.load(self.cache_path)
                # Verify cache consistency
                if len(images) == len(self.df):
                    print(f"Loaded {len(images)} images from cache: {self.cache_path}")
                    return images
                else:
                    print(
                        f"Cache mismatch (Cache: {len(images)}, DF: {len(self.df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache {self.cache_path}: {e}. Recomputing...")

        # 2. Compute from scratch
        print(
            f"Processing {len(self.df)} images for {os.path.basename(self.cache_path)}..."
        )
        images = np.zeros((len(self.df), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        for idx, row in self.df.iterrows():
            file_path = os.path.join(INPUT_DIR, row["file_path"])

            # Read image
            img = cv2.imread(file_path)
            if img is None:
                # Handle missing image (should not happen if metadata is verified)
                img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            images[idx] = img

        # 3. Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.save(self.cache_path, images)
        print(f"Saved cache to {self.cache_path}")

        return images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Handle Label
        label_str = self.df.iloc[idx]["Id"]

        target = -1
        # If we have an encoder and the label is valid
        if self.label_encoder is not None and pd.notna(label_str):
            # Returns -1 for 'new_whale' or unknown classes
            target = self.label_encoder.get(label_str, -1)

        return image, target


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader, label_encoder
    """
    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # 2. Filter Training Data
    # Strictly filter out 'new_whale' for ArcFace training
    train_df_filtered = train_df[train_df["Id"] != "new_whale"].copy()

    # 3. Handle Debug Mode
    if DEBUG:
        train_df_filtered = train_df_filtered.iloc[:DEBUG_SIZE]
        val_df = val_df.iloc[:DEBUG_SIZE]
        test_df = test_df.iloc[:DEBUG_SIZE]

        # Use debug specific cache names to avoid conflicts
        train_cache = f"train_images_debug_{IMAGE_SIZE}.npy"
        val_cache = f"val_images_debug_{IMAGE_SIZE}.npy"
        test_cache = f"test_images_debug_{IMAGE_SIZE}.npy"
    else:
        train_cache = f"train_images_{IMAGE_SIZE}.npy"
        val_cache = f"val_images_{IMAGE_SIZE}.npy"
        test_cache = f"test_images_{IMAGE_SIZE}.npy"

    # 4. Create Label Encoder
    # Map known IDs to integers 0..N-1
    unique_ids = sorted(train_df_filtered["Id"].unique())
    label_encoder = {label: idx for idx, label in enumerate(unique_ids)}

    # 5. Define Transforms
    train_transforms = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=IMAGE_SIZE // 8,
                max_width=IMAGE_SIZE // 8,
                min_holes=1,
                fill_value=0,
                p=0.5,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 6. Instantiate Datasets
    train_dataset = WhaleDataset(
        train_df_filtered,
        transforms=train_transforms,
        cache_name=train_cache,
        load_cache=load_cached_data,
        label_encoder=label_encoder,
    )

    val_dataset = WhaleDataset(
        val_df,
        transforms=val_transforms,
        cache_name=val_cache,
        load_cache=load_cached_data,
        label_encoder=label_encoder,
    )

    test_dataset = WhaleDataset(
        test_df,
        transforms=val_transforms,
        cache_name=test_cache,
        load_cache=load_cached_data,
        label_encoder=label_encoder,
    )

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_encoder
