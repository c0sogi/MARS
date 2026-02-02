import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles 128x128x2 inputs (Grayscale Image + Depth Channel).
    """

    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (128, 128, 2)

        if self.masks is not None:
            mask = self.masks[idx]  # Shape: (128, 128, 1)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # Fallback if no transform provided (convert to tensor)
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask.transpose(2, 0, 1))

            return image, mask
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1))

            # Return a placeholder for mask to keep signature consistent or just image
            # Based on model.py inference loop, it expects (images, _)
            return image, ""


def preprocess_and_cache(metadata_path, cache_dir, load_cached_data=True, mode="train"):
    """
    Loads images, pre-processes them (pad, depth fusion), and caches as .npy.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_dir (str): Directory to store cached .npy files.
        load_cached_data (bool): Whether to attempt loading from cache.
        mode (str): 'train', 'val', or 'test'. Used for cache filenames.

    Returns:
        tuple: (ids, images, masks, coverage_classes)
               masks is None if mode=='test'.
               coverage_classes is None if column not in metadata.
    """
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")
    images_path = os.path.join(cache_dir, f"{mode}_images.npy")
    masks_path = os.path.join(cache_dir, f"{mode}_masks.npy")
    classes_path = os.path.join(cache_dir, f"{mode}_classes.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        # Check if masks exist (required for non-test modes)
        if mode == "test" or os.path.exists(masks_path):
            print(f"Loading cached {mode} data from {cache_dir}...")
            ids = np.load(ids_path, allow_pickle=True)
            images = np.load(images_path)

            masks = None
            if mode != "test":
                masks = np.load(masks_path)

            # Try loading classes if they exist
            coverage_classes = None
            if os.path.exists(classes_path):
                coverage_classes = np.load(classes_path)

            return ids, images, masks, coverage_classes

    print(f"Processing {mode} data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    mask_list = []
    id_list = []
    class_list = []

    # Pre-calculate depth stats for normalization (Global min/max from analysis)
    DEPTH_MIN = 51.0
    DEPTH_MAX = 959.0

    input_dir = "./input"

    for idx, row in df.iterrows():
        id_ = row["id"]
        z = row["z"]

        # Load Image
        # Use relative path from metadata
        img_path = os.path.join(input_dir, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            continue

        # Normalize Image
        img = img.astype(np.float32) / 255.0

        # Load Mask if not test
        mask = None
        if mode != "test":
            mask_path = os.path.join(input_dir, row["mask_path"])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"Warning: Could not load mask {mask_path}. Skipping.")
                continue
            mask = mask_img.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # Reflection Padding 101 -> 128
        # Pad: (128-101) = 27. Top=13, Bottom=14. Left=13, Right=14.
        pad_h = 128 - 101
        pad_w = 128 - 101
        p_top = pad_h // 2
        p_bot = pad_h - p_top
        p_left = pad_w // 2
        p_right = pad_w - p_left

        img_padded = cv2.copyMakeBorder(
            img, p_top, p_bot, p_left, p_right, cv2.BORDER_REFLECT_101
        )

        if mask is not None:
            mask_padded = cv2.copyMakeBorder(
                mask, p_top, p_bot, p_left, p_right, cv2.BORDER_REFLECT_101
            )
            mask_list.append(mask_padded)

        # Depth Channel
        z_norm = (z - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        depth_channel = np.full_like(img_padded, z_norm)

        # Stack: (128, 128, 2)
        combined = np.stack([img_padded, depth_channel], axis=-1)

        img_list.append(combined)
        id_list.append(id_)

        if "coverage_class" in row:
            class_list.append(row["coverage_class"])

    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list)

    np.save(ids_path, ids)
    np.save(images_path, images)

    masks = None
    if mode != "test":
        masks = np.array(mask_list, dtype=np.float32)
        # Expand dims for masks: (N, 128, 128) -> (N, 128, 128, 1)
        masks = masks[..., np.newaxis]
        np.save(masks_path, masks)

    coverage_classes = None
    if class_list:
        coverage_classes = np.array(class_list)
        np.save(classes_path, coverage_classes)

    return ids, images, masks, coverage_classes


def get_loaders(
    train_metadata_path="./metadata/train.csv",
    val_metadata_path="./metadata/val.csv",
    cache_dir="./working/idea_10",
    batch_size=32,
    num_workers=4,
    load_cached_data=True,
    debug=False,
):
    """
    Generates Train and Validation DataLoaders using a 90/10 Stratified Split.
    Combines data from train.csv and val.csv and re-splits.
    """
    set_seed(42)

    # 1. Load/Process Data from both source files
    # We use distinct mode names 'train_src' and 'val_src' to avoid cache collision
    # if we were to change logic, but here we map to the files provided.
    ids_1, imgs_1, masks_1, classes_1 = preprocess_and_cache(
        train_metadata_path, cache_dir, load_cached_data, mode="train_src"
    )
    ids_2, imgs_2, masks_2, classes_2 = preprocess_and_cache(
        val_metadata_path, cache_dir, load_cached_data, mode="val_src"
    )

    # 2. Concatenate datasets
    all_images = np.concatenate([imgs_1, imgs_2], axis=0)
    all_masks = np.concatenate([masks_1, masks_2], axis=0)
    all_classes = np.concatenate([classes_1, classes_2], axis=0)

    # 3. Perform 90/10 Stratified Split
    # We split indices to keep arrays aligned
    indices = np.arange(len(all_images))

    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=all_classes
    )

    train_images = all_images[train_idx]
    train_masks = all_masks[train_idx]

    val_images = all_images[val_idx]
    val_masks = all_masks[val_idx]

    if debug:
        print("Debug mode: Reducing dataset size.")
        train_images = train_images[:64]
        train_masks = train_masks[:64]
        val_images = val_images[:32]
        val_masks = val_masks[:32]

    print(f"Train Set: {len(train_images)} images")
    print(f"Val Set: {len(val_images)} images")

    # 4. Define Transforms
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), ToTensorV2()])

    val_transform = A.Compose([ToTensorV2()])

    # 5. Create Datasets and Loaders
    train_dataset = SaltDataset(train_images, train_masks, transform=train_transform)
    val_dataset = SaltDataset(val_images, val_masks, transform=val_transform)

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


def get_test_loader(
    test_metadata_path="./metadata/test.csv",
    cache_dir="./working/idea_10",
    batch_size=32,
    num_workers=4,
    load_cached_data=True,
):
    """
    Generates Test DataLoader.
    """
    ids, images, _, _ = preprocess_and_cache(
        test_metadata_path, cache_dir, load_cached_data, mode="test"
    )

    test_transform = A.Compose([ToTensorV2()])

    test_dataset = SaltDataset(images, masks=None, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, ids
