import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)


def get_transforms(split: str):
    """
    Returns the transformations for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transformations.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # ToTensor converts [0, 255] -> [0.0, 1.0].
                # We strictly adhere to the "normalized to [0, 1] range" requirement
                # and avoid further standardization (mean/std subtraction).
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


def load_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata, utilizing caching for images and labels.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_npy, labels_npy, ids_npy)
    """
    suffix = "_debug" if DEBUG else ""
    cache_img_path = os.path.join(WORKING_DIR, f"{cache_prefix}_images{suffix}.npy")
    cache_lbl_path = os.path.join(WORKING_DIR, f"{cache_prefix}_labels{suffix}.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"{cache_prefix}_ids{suffix}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_img_path)
            and os.path.exists(cache_lbl_path)
            and os.path.exists(cache_ids_path)
        ):
            # print(f"Loading {cache_prefix} data from cache...")
            images = np.load(cache_img_path)
            labels = np.load(cache_lbl_path)
            ids = np.load(cache_ids_path)
            return images, labels, ids

    # 2. Process from scratch
    print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    if DEBUG:
        df = df.head(DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: sampled {len(df)} rows for {cache_prefix}.")

    img_list = []
    lbl_list = []
    id_list = []

    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure size (though dataset is fixed 32x32)
        if img.shape[:2] != IMAGE_SIZE:
            img = cv2.resize(img, IMAGE_SIZE)

        img_list.append(img)
        lbl_list.append(row["has_cactus"])
        id_list.append(row["id"])

    images = np.array(
        img_list, dtype=np.uint8
    )  # Keep as uint8 to save space until transform
    labels = np.array(lbl_list, dtype=np.float32)
    ids = np.array(id_list)

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    np.save(cache_img_path, images)
    np.save(cache_lbl_path, labels)
    np.save(cache_ids_path, ids)
    print(f"Saved {cache_prefix} data to cache.")

    return images, labels, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is HWC, uint8
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        # Return label as float tensor
        return img, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_imgs, train_lbls, train_ids = load_data(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_data(VAL_METADATA_PATH, "val", load_cached_data)
    test_imgs, test_lbls, test_ids = load_data(
        TEST_METADATA_PATH, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader
