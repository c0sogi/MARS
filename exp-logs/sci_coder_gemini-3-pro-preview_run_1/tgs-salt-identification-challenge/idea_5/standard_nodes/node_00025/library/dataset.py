import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.utils import rle_decode

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_5"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles images with fused depth channels and binary masks.
    """

    def __init__(self, images, masks=None, ids=None, transform=None):
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, 2) - Channel 0: Seismic, Channel 1: Depth
        image = self.images[idx]

        data = {"image": image}
        if self.masks is not None:
            mask = self.masks[idx]
            data["mask"] = mask

        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if "mask" in data:
                mask = augmented["mask"]
                # Ensure mask is (1, H, W) float tensor
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                mask = mask.float()
        else:
            # Fallback if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if "mask" in data:
                mask = torch.from_numpy(data["mask"]).unsqueeze(0).float()

        if self.masks is not None:
            return image, mask, self.ids[idx]
        else:
            return image, self.ids[idx]


def pad_image(img, target_size=128):
    """
    Applies Reflection Padding to resize image to target_size.
    """
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w

    if pad_h < 0 or pad_w < 0:
        return cv2.resize(img, (target_size, target_size))

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Reflection padding (works for multi-channel images too)
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )
    return padded


def load_and_preprocess(df, data_type, cache_dir, load_cached, depth_stats=None):
    """
    Loads images, processes them (depth fusion, padding), and caches them as .npy files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    images_path = os.path.join(cache_dir, f"{data_type}_images.npy")
    masks_path = os.path.join(cache_dir, f"{data_type}_masks.npy")
    ids_path = os.path.join(cache_dir, f"{data_type}_ids.npy")

    # Determine if we should look for masks
    # Test set might have 'rle_mask' column but it's dummy data, so we ignore it for 'test' type
    has_masks = (
        (data_type != "test")
        and ("rle_mask" in df.columns)
        and (not df["rle_mask"].isnull().all())
    )

    # 1. Try to load cached data
    if load_cached:
        if os.path.exists(images_path) and os.path.exists(ids_path):
            # If masks are expected, check if they exist
            if has_masks and not os.path.exists(masks_path):
                pass  # Cache invalid/incomplete, recompute
            else:
                print(f"Loading cached {data_type} data from {cache_dir}...")
                images = np.load(images_path)
                ids = np.load(ids_path)
                masks = np.load(masks_path) if has_masks else None
                return images, masks, ids

    # 2. Compute from scratch
    print(f"Processing {data_type} data from scratch...")

    img_list = []
    mask_list = []
    id_list = []

    # Use provided depth stats or calculate from current df (fallback)
    if depth_stats is None:
        z_mean = df["z"].mean()
        z_std = df["z"].std()
    else:
        z_mean, z_std = depth_stats

    for idx, row in df.iterrows():
        img_id = row["id"]
        # Metadata paths are relative, e.g., "train/images/xxxx.png"
        img_path = os.path.join(INPUT_DIR, row["image_path"])

        # Load Image (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        # Normalize Image to 0-1
        img = img.astype(np.float32) / 255.0

        # Depth Channel Fusion
        z = row["z"]
        # Normalize depth
        z_norm = (z - z_mean) / (z_std + 1e-8)

        # Create Depth Map with same spatial dimensions
        depth_map = np.full_like(img, z_norm)

        # Stack: (H, W, 2)
        img_fused = np.dstack([img, depth_map])

        # Reflection Padding to 128x128
        img_padded = pad_image(img_fused, target_size=IMG_SIZE_TARGET)
        img_list.append(img_padded)

        # Process Mask if available
        if has_masks:
            rle = row["rle_mask"]
            mask = rle_decode(rle, shape=(IMG_SIZE_ORIG, IMG_SIZE_ORIG))

            # Pad mask (nearest neighbor implicitly via copyMakeBorder for binary)
            mask_padded = pad_image(mask, target_size=IMG_SIZE_TARGET)

            # Ensure binary and float
            mask_padded = (mask_padded > 0.5).astype(np.float32)
            mask_list.append(mask_padded)

        id_list.append(img_id)

    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list)
    masks = np.array(mask_list, dtype=np.float32) if has_masks else None

    # Save to cache
    print(f"Saving {data_type} data to cache...")
    np.save(images_path, images)
    np.save(ids_path, ids)
    if masks is not None:
        np.save(masks_path, masks)

    return images, masks, ids


def get_dataloaders(
    train_csv_path="./metadata/train.csv",
    val_csv_path="./metadata/val.csv",
    test_csv_path="./metadata/test.csv",
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    debug=False,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path) if os.path.exists(test_csv_path) else None

    if debug:
        print("Debug mode: reducing dataset size.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        if df_test is not None:
            df_test = df_test.head(50)

    # Compute Global Depth Stats from Training Set
    # This ensures consistent normalization across all sets
    z_mean = df_train["z"].mean()
    z_std = df_train["z"].std()
    depth_stats = (z_mean, z_std)

    # Process Data
    train_imgs, train_masks, train_ids = load_and_preprocess(
        df_train, "train", CACHE_DIR, load_cached_data, depth_stats
    )
    val_imgs, val_masks, val_ids = load_and_preprocess(
        df_val, "val", CACHE_DIR, load_cached_data, depth_stats
    )

    test_imgs, test_masks, test_ids = None, None, None
    if df_test is not None:
        test_imgs, _, test_ids = load_and_preprocess(
            df_test, "test", CACHE_DIR, load_cached_data, depth_stats
        )

    # Define Transforms
    # Train: Horizontal Flip + ToTensor
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), ToTensorV2()])

    # Val/Test: ToTensor only
    val_transform = A.Compose([ToTensorV2()])

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs, train_masks, train_ids, transform=train_transform
    )
    val_dataset = SaltDataset(val_imgs, val_masks, val_ids, transform=val_transform)

    test_dataset = None
    if test_imgs is not None:
        test_dataset = SaltDataset(test_imgs, None, test_ids, transform=val_transform)

    # Create DataLoaders
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
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
