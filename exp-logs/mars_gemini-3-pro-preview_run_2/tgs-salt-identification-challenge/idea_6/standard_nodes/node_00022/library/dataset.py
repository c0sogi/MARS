import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# Constants
INPUT_ROOT = "./input"
CACHE_DIR = "./working/idea_6/"
IMG_SIZE = 101
TARGET_SIZE = 128


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


class SaltDataset(Dataset):
    def __init__(self, images, depths, ids, masks=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W) or (N, H, W, C).
            depths (np.ndarray): Array of depth values (N,).
            ids (np.ndarray): Array of IDs (N,).
            masks (np.ndarray, optional): Array of masks (N, H, W).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]
        id_ = self.ids[idx]

        # Ensure image has channel dimension for Albumentations if grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            # Ensure mask has channel dimension
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Convert depth to tensor float
        depth = torch.tensor([depth], dtype=torch.float32)

        if mask is not None:
            # Mask is typically float for BCE/Dice loss
            mask = mask.float()
            return image, mask, depth, id_
        else:
            return image, depth, id_


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    transforms_list = []

    # 1. Pad to 128x128 using reflection
    transforms_list.append(
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            always_apply=True,
        )
    )

    if phase == "train":
        # 2. Augmentations (Elastic, ShiftScaleRotate, Flip)
        transforms_list.extend(
            [
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

    # 3. Normalization & Tensor Conversion
    # Using standard ImageNet stats or 0.5/0.5 for grayscale
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ]
    )

    return A.Compose(transforms_list)


def process_images_from_metadata(df, input_root):
    """
    Reads images and masks based on metadata DataFrame.
    """
    images = []
    masks = []
    ids = []
    depths = []

    has_masks = "mask_path" in df.columns

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(input_root, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load Mask if exists
        if has_masks:
            mask_path = os.path.join(input_root, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            # Binarize mask (0 or 255 -> 0 or 1)
            mask = (mask > 127).astype(np.uint8)
            masks.append(mask)

        images.append(img)
        ids.append(row["id"])
        depths.append(row["z"])

    images = np.array(images)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    if has_masks:
        masks = np.array(masks)
        return images, masks, depths, ids
    else:
        return images, None, depths, ids


def prepare_data(load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    Returns a dictionary with 'train', 'val', 'test' keys containing data arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    data_store = {}

    meta_files = {
        "train": "./metadata/train.csv",
        "val": "./metadata/val.csv",
        "test": "./metadata/test.csv",
    }

    for split in splits:
        # Define cache paths
        cache_img = os.path.join(CACHE_DIR, f"{split}_images.npy")
        cache_mask = os.path.join(CACHE_DIR, f"{split}_masks.npy")
        cache_depth = os.path.join(CACHE_DIR, f"{split}_depths.npy")
        cache_id = os.path.join(CACHE_DIR, f"{split}_ids.npy")

        # Check existence
        is_cached = (
            os.path.exists(cache_img)
            and os.path.exists(cache_depth)
            and os.path.exists(cache_id)
        )
        if split in ["train", "val"]:
            is_cached = is_cached and os.path.exists(cache_mask)

        if load_cached_data and is_cached:
            # Load from cache
            images = np.load(cache_img)
            depths = np.load(cache_depth)
            ids = np.load(cache_id)
            masks = None
            if split in ["train", "val"]:
                masks = np.load(cache_mask)
        else:
            # Process from scratch
            df = pd.read_csv(meta_files[split])
            images, masks, depths, ids = process_images_from_metadata(df, INPUT_ROOT)

            # Save to cache
            np.save(cache_img, images)
            np.save(cache_depth, depths)
            np.save(cache_id, ids)
            if masks is not None:
                np.save(cache_mask, masks)

        data_store[split] = {
            "images": images,
            "masks": masks,
            "depths": depths,
            "ids": ids,
        }

    return data_store


def get_dataloaders(data_store, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for train, val, and test.
    Performs depth normalization based on training statistics.
    """
    # Calculate depth stats from training set
    train_depths = data_store["train"]["depths"]
    # Filter out NaNs if any (though train shouldn't have them)
    valid_train_depths = train_depths[~np.isnan(train_depths)]

    mean_depth = np.mean(valid_train_depths)
    std_depth = np.std(valid_train_depths)

    # Apply normalization to all splits
    # Note: This modifies the arrays in the dictionary in-place or copies them.
    # We use copies to avoid polluting the original data if called multiple times.
    norm_depths = {}
    for split in ["train", "val", "test"]:
        d = data_store[split]["depths"].copy()
        # Handle NaNs in test if they exist (will remain NaN, handled by imputation later if needed)
        # But assuming imputation is done before this if needed, or we just normalize what we have.
        d = (d - mean_depth) / (std_depth + 1e-8)
        norm_depths[split] = d

    # Create Datasets
    train_ds = SaltDataset(
        data_store["train"]["images"],
        norm_depths["train"],
        data_store["train"]["ids"],
        data_store["train"]["masks"],
        transform=get_transforms("train"),
    )

    val_ds = SaltDataset(
        data_store["val"]["images"],
        norm_depths["val"],
        data_store["val"]["ids"],
        data_store["val"]["masks"],
        transform=get_transforms("val"),
    )

    test_ds = SaltDataset(
        data_store["test"]["images"],
        norm_depths["test"],
        data_store["test"]["ids"],
        data_store["test"]["masks"],  # Usually None
        transform=get_transforms("test"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
