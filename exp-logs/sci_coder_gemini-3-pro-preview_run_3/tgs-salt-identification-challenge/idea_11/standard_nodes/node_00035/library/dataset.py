import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import rle_decode

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_11"
ORIG_SIZE = 101
TARGET_SIZE = 128

# Global depth stats for normalization
DEPTH_MIN = 0.0
DEPTH_MAX = 1000.0


class SaltDataset(Dataset):
    def __init__(self, images, depths, ids, masks=None, transform=None):
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (H, W)
        image = self.images[idx]

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Load and normalize depth
        depth_val = self.depths[idx]
        depth_norm = (depth_val - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        # Create depth channel
        depth_channel = np.full_like(image, depth_norm, dtype=np.float32)

        # Construct 3-channel input: [Seismic, Seismic, Depth]
        image_3c = np.dstack([image, image, depth_channel])

        mask = None
        if self.masks is not None:
            mask = self.masks[idx].astype(np.float32)

        # Apply transforms (Augmentation + Padding)
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image_3c, mask=mask)
                image_3c = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image_3c)
                image_3c = augmented["image"]

        data = {"image": image_3c, "id": self.ids[idx]}

        if mask is not None:
            # Ensure mask is (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            data["mask"] = mask

        return data


def load_or_create_data(mode="train", load_cached_data=True, debug=False):
    """
    Loads data from cache or creates it from source metadata.
    mode: 'train' (merges train+val) or 'test'
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "images": os.path.join(CACHE_DIR, f"cached_{mode}_images.npy"),
        "masks": os.path.join(CACHE_DIR, f"cached_{mode}_masks.npy"),
        "depths": os.path.join(CACHE_DIR, f"cached_{mode}_depths.npy"),
        "ids": os.path.join(CACHE_DIR, f"cached_{mode}_ids.npy"),
        "classes": os.path.join(CACHE_DIR, f"cached_{mode}_coverage_classes.npy"),
    }

    # Check if cache exists
    cache_exists = all(
        os.path.exists(f)
        for k, f in cache_files.items()
        if k != "masks" or mode == "train"
    )
    if mode == "test":
        cache_exists = (
            os.path.exists(cache_files["images"])
            and os.path.exists(cache_files["depths"])
            and os.path.exists(cache_files["ids"])
        )

    if load_cached_data and cache_exists:
        images = np.load(cache_files["images"])
        depths = np.load(cache_files["depths"])
        ids = np.load(cache_files["ids"])

        masks = None
        classes = None

        if mode == "train":
            masks = np.load(cache_files["masks"])
            classes = np.load(cache_files["classes"])

        if debug:
            limit = 100
            images = images[:limit]
            depths = depths[:limit]
            ids = ids[:limit]
            if masks is not None:
                masks = masks[:limit]
            if classes is not None:
                classes = classes[:limit]

        return images, masks, depths, ids, classes

    # Load Metadata
    if mode == "train":
        df_train = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
        df_val = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.zeros((n_samples, ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
    depths = np.zeros(n_samples, dtype=np.float32)
    ids = []

    masks = None
    classes = None
    if mode == "train":
        masks = np.zeros((n_samples, ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
        classes = np.zeros(n_samples, dtype=np.int32)

    # Iterate and load
    for i, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
        images[i] = img

        # Load Depth
        depths[i] = row["z"]

        # Store ID
        ids.append(row["id"])

        if mode == "train":
            # Decode Mask
            rle = row["rle_mask"]
            mask = rle_decode(rle)
            masks[i] = mask

            # Store Class
            classes[i] = row["coverage_class"]

    ids = np.array(ids)

    # Save to cache
    np.save(cache_files["images"], images)
    np.save(cache_files["depths"], depths)
    np.save(cache_files["ids"], ids)

    if mode == "train":
        np.save(cache_files["masks"], masks)
        np.save(cache_files["classes"], classes)

    if debug:
        limit = 100
        images = images[:limit]
        depths = depths[:limit]
        ids = ids[:limit]
        if masks is not None:
            masks = masks[:limit]
        if classes is not None:
            classes = classes[:limit]

    return images, masks, depths, ids, classes


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the given phase.
    """
    transforms = []

    if phase == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5
                ),
            ]
        )

    # Padding is applied in both train and val/test to reach 128x128
    transforms.append(
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            always_apply=True,
        )
    )

    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_dataloaders(
    fold_idx=0,
    n_folds=5,
    batch_size=32,
    load_cached_data=True,
    num_workers=2,
    debug=False,
):
    """
    Creates train and validation dataloaders for a specific fold.
    """
    # Load all training data
    images, masks, depths, ids, classes = load_or_create_data(
        mode="train", load_cached_data=load_cached_data, debug=debug
    )

    # Stratified Split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Dummy X, we split based on classes
    dummy_X = np.zeros(len(classes))
    splits = list(skf.split(dummy_X, classes))

    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_dataset = SaltDataset(
        images[train_idx],
        depths[train_idx],
        ids[train_idx],
        masks[train_idx],
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        images[val_idx],
        depths[val_idx],
        ids[val_idx],
        masks[val_idx],
        transform=get_transforms("val"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, load_cached_data=True, num_workers=2, debug=False):
    """
    Creates test dataloader.
    """
    images, _, depths, ids, _ = load_or_create_data(
        mode="test", load_cached_data=load_cached_data, debug=debug
    )

    test_dataset = SaltDataset(
        images, depths, ids, masks=None, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
