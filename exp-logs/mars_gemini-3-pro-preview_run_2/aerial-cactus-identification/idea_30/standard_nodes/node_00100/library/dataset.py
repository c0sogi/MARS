import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    NUM_WORKERS,
    IMG_SIZE,
)


def get_transforms(split="train"):
    """
    Returns the transformation pipeline for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        # Light augmentation: Flips only, followed by normalization to [0, 1] via ToTensor
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # Validation/Test: Only normalization to [0, 1]
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


class CactusDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split_name,
        transform=None,
        load_cached_data=True,
        limit=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            split_name (str): 'train', 'val', or 'test' (used for cache naming).
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to load data from cache if available.
            limit (int, optional): Limit the dataset size (for debugging).
        """
        self.transform = transform
        self.split_name = split_name
        self.metadata_path = metadata_path
        self.cache_dir = OUTPUT_DIR

        # Define cache file paths
        self.images_cache = os.path.join(self.cache_dir, f"{split_name}_images.npy")
        self.labels_cache = os.path.join(self.cache_dir, f"{split_name}_labels.npy")
        self.ids_cache = os.path.join(self.cache_dir, f"{split_name}_ids.npy")

        # Load data (either from cache or raw files)
        self.images, self.labels, self.ids = self._load_data(load_cached_data)

        # Apply limit if requested (for debugging/testing)
        if limit is not None:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]
            self.ids = self.ids[:limit]

    def _load_data(self, load_cached_data):
        """
        Internal method to handle caching logic.
        """
        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(self.images_cache)
                and os.path.exists(self.labels_cache)
                and os.path.exists(self.ids_cache)
            ):
                try:
                    images = np.load(self.images_cache)
                    labels = np.load(self.labels_cache)
                    ids = np.load(self.ids_cache)
                    return images, labels, ids
                except Exception as e:
                    print(
                        f"Failed to load cache for {self.split_name}: {e}. Recomputing..."
                    )

        # 2. Compute from scratch
        df = pd.read_csv(self.metadata_path)

        images_list = []
        labels_list = []
        ids_list = []

        for idx, row in df.iterrows():
            rel_path = row["file_path"]
            # Ensure path is correct relative to INPUT_DIR
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Load image
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images_list.append(img)
            labels_list.append(row["has_cactus"])
            ids_list.append(row["id"])

        images = np.array(images_list, dtype=np.uint8)
        labels = np.array(labels_list, dtype=np.float32)
        ids = np.array(ids_list)

        # 3. Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.images_cache, images)
        np.save(self.labels_cache, labels)
        np.save(self.ids_cache, ids)

        return images, labels, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        if self.transform:
            # transform expects PIL Image or Tensor or ndarray
            # ToTensor handles HWC uint8 ndarray -> CHW float32 tensor [0,1]
            img = self.transform(img)

        return img, label, img_id


def get_loaders(batch_size, load_cached_data=True, limit=None):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached data.
        limit (int, optional): Limit dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    train_dataset = CactusDataset(
        metadata_path=TRAIN_METADATA_PATH,
        split_name="train",
        transform=train_transform,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    val_dataset = CactusDataset(
        metadata_path=VAL_METADATA_PATH,
        split_name="val",
        transform=val_transform,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    test_dataset = CactusDataset(
        metadata_path=TEST_METADATA_PATH,
        split_name="test",
        transform=val_transform,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Avoid small batches affecting BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
