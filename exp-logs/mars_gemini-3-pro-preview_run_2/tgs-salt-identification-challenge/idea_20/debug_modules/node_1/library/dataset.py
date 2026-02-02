import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Constants
CACHE_DIR = "./working/idea_20/"
INPUT_DIR = "./input"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128
# ImageNet stats averaged for grayscale: (0.485+0.456+0.406)/3, (0.229+0.224+0.225)/3
IMAGENET_MEAN = [0.449]
IMAGENET_STD = [0.226]


def pad_image(img, target_size=IMG_SIZE_TARGET):
    """Pads image from 101x101 to 128x128 using reflection."""
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w

    if pad_h < 0 or pad_w < 0:
        return cv2.resize(img, (target_size, target_size))

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def preprocess_and_cache(df, mode, load_cached_data=True):
    """
    Loads images/masks, pads them, and caches as numpy arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache filenames
    f_images = os.path.join(CACHE_DIR, f"{mode}_images.npy")
    f_masks = os.path.join(CACHE_DIR, f"{mode}_masks.npy")
    f_depths = os.path.join(CACHE_DIR, f"{mode}_depths.npy")
    f_ids = os.path.join(CACHE_DIR, f"{mode}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(f_images)
            and os.path.exists(f_ids)
            and os.path.exists(f_depths)
        ):
            # Check mask existence only if not test mode
            if mode == "test" or os.path.exists(f_masks):
                print(f"Loading {mode} data from cache...")
                images = np.load(f_images)
                depths = np.load(f_depths)
                ids = np.load(f_ids)
                masks = np.load(f_masks) if mode != "test" else None
                return images, masks, depths, ids

    # 2. Process from scratch
    print(f"Processing {mode} data from scratch...")

    img_list = []
    mask_list = []
    depth_list = []
    id_list = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Pad Image
        img = pad_image(img)
        img_list.append(img)

        # Load Mask (if exists)
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = os.path.join(INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Pad Mask
            mask = pad_image(mask)
            # Binarize (0 or 1)
            mask = (mask > 127).astype(np.uint8)
            mask_list.append(mask)

        # Depth
        depth_list.append(row["z"])
        id_list.append(row["id"])

    # Convert to numpy arrays
    # Images: (N, 128, 128)
    images = np.array(img_list, dtype=np.uint8)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    if len(mask_list) > 0:
        masks = np.array(mask_list, dtype=np.uint8)
    else:
        masks = None

    # Save to cache
    np.save(f_images, images)
    np.save(f_depths, depths)
    np.save(f_ids, ids)
    if masks is not None:
        np.save(f_masks, masks)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        ids,
        transform=None,
        depth_stats=None,
        pseudo_labels=None,
    ):
        """
        Args:
            images: (N, H, W) uint8
            masks: (N, H, W) uint8 or None
            depths: (N,) float32
            ids: (N,) string
            transform: Albumentations transform
            depth_stats: (mean, std) for depth normalization
            pseudo_labels: dict {id: soft_mask_array} where soft_mask is (101, 101)
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.depth_stats = depth_stats if depth_stats else (0.0, 1.0)
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (128, 128)
        img_id = self.ids[idx]
        depth = self.depths[idx]

        # Determine target (Mask or Pseudo-Label)
        mask = None

        # Check for pseudo-label first
        if self.pseudo_labels is not None and img_id in self.pseudo_labels:
            # Pseudo label is likely 101x101, need to pad
            p_mask = self.pseudo_labels[img_id]
            mask = pad_image(p_mask)
            # Ensure float32 for soft targets
            mask = mask.astype(np.float32)
        elif self.masks is not None:
            mask = self.masks[idx].astype(np.float32)

        # Augmentations
        if self.transform:
            # Albumentations expects HWC
            # Add dummy channel for grayscale if needed, or just pass HW
            # Since we are using ToTensorV2, it handles shape, but typically A expects numpy

            # If mask is present
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Depth Normalization
        d_mean, d_std = self.depth_stats
        depth = (depth - d_mean) / (d_std + 1e-6)

        # If mask is None (test set without pseudo labels), return dummy
        if mask is None:
            mask = torch.zeros((1, 128, 128), dtype=torch.float32)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1, 128, 128)

        # Depth to tensor
        depth = torch.tensor([depth], dtype=torch.float32)

        return img, mask, depth, img_id


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # Elastic Transform parameters from "Idea"
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
        )


def get_loaders(batch_size=32, debug=False, load_cached_data=True, pseudo_labels=None):
    """
    Main function to get data loaders.

    Args:
        batch_size (int): Batch size.
        debug (bool): If True, subsample data.
        load_cached_data (bool): Whether to use cached numpy arrays.
        pseudo_labels (dict): Dictionary of {id: soft_mask} for self-training.
    """
    set_seed(42)

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # Calculate Depth Stats from Training Data (Raw)
    # We calculate this before processing/caching to ensure consistency
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std()
    depth_stats = (depth_mean, depth_std)

    # Process/Load Data
    train_imgs, train_masks, train_depths, train_ids = preprocess_and_cache(
        train_df, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = preprocess_and_cache(
        val_df, "val", load_cached_data
    )
    test_imgs, _, test_depths, test_ids = preprocess_and_cache(
        test_df, "test", load_cached_data
    )

    # Create Datasets
    train_ds = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transform=get_transforms("train"),
        depth_stats=depth_stats,
        pseudo_labels=pseudo_labels,
    )

    val_ds = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transform=get_transforms("val"),
        depth_stats=depth_stats,
    )

    test_ds = SaltDataset(
        test_imgs,
        None,
        test_depths,
        test_ids,
        transform=get_transforms("test"),
        depth_stats=depth_stats,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(
        f"Data Loaders created. Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
