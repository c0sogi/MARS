import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed

# Set random seed for reproducibility
set_seed(42)


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles on-the-fly padding, normalization, and depth channel integration.
    """

    def __init__(self, images, masks, depths, ids, depth_mean, depth_std, mode="train"):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load raw data
        img = self.images[idx]  # Shape: (101, 101), dtype: uint8
        depth_val = self.depths[idx]
        image_id = self.ids[idx]

        # 1. Depth Normalization
        # Normalize depth using training set statistics
        z = (depth_val - self.depth_mean) / (self.depth_std + 1e-8)

        # 2. Padding (101x101 -> 128x128) using Reflection
        # Calculate padding amounts to center the image
        target_size = 128
        h, w = img.shape
        pad_h = target_size - h
        pad_w = target_size - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )

        # 3. Normalization (0-1)
        img_float = img_padded.astype(np.float32) / 255.0

        # 4. Depth Channel Creation
        # Repeat scalar depth to match spatial dimensions (Input Fusion)
        depth_map = np.full((target_size, target_size), z, dtype=np.float32)

        # 5. Mask Processing (if available)
        mask_float = None
        if self.masks is not None:
            mask = self.masks[idx]
            mask_padded = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
            )
            # Binarize and convert to float (0.0 or 1.0)
            mask_float = (mask_padded > 127).astype(np.float32)

        # 6. Augmentation (Random Horizontal Flip) - Only for training
        if self.mode == "train":
            if np.random.rand() > 0.5:
                img_float = np.flip(img_float, axis=1).copy()
                depth_map = np.flip(depth_map, axis=1).copy()
                if mask_float is not None:
                    mask_float = np.flip(mask_float, axis=1).copy()

        # 7. Stack Channels -> (2, 128, 128)
        # Channel 0: Image, Channel 1: Depth
        input_tensor = np.stack([img_float, depth_map], axis=0)
        input_tensor = torch.from_numpy(input_tensor).float()

        if self.mode in ["train", "val"]:
            # Return (Input, Mask, ID)
            # Mask shape: (1, 128, 128)
            mask_tensor = torch.from_numpy(mask_float).float().unsqueeze(0)
            return input_tensor, mask_tensor, image_id
        else:
            # Return (Input, ID) for inference
            return input_tensor, image_id


def load_and_process_data(
    metadata_path, mode, load_cached_data=True, cache_dir="./working/idea_1"
):
    """
    Loads images and metadata, using caching to speed up subsequent runs.
    Reads raw images, converts to numpy arrays, and saves/loads from cache directory.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    cache_files = {
        "images": os.path.join(cache_dir, f"{mode}_images.npy"),
        "masks": os.path.join(cache_dir, f"{mode}_masks.npy"),
        "depths": os.path.join(cache_dir, f"{mode}_depths.npy"),
        "ids": os.path.join(cache_dir, f"{mode}_ids.npy"),
    }

    # Check if cache exists
    # Note: Test set doesn't have masks, so we check accordingly
    required_keys = ["images", "depths", "ids"]
    if mode != "test":
        required_keys.append("masks")

    cache_exists = all(os.path.exists(cache_files[k]) for k in required_keys)

    if load_cached_data and cache_exists:
        print(f"Loading {mode} data from cache ({cache_dir})...")
        images = np.load(cache_files["images"])
        depths = np.load(cache_files["depths"])
        ids = np.load(cache_files["ids"])
        masks = np.load(cache_files["masks"]) if mode != "test" else None
        return images, masks, depths, ids

    print(f"Processing {mode} data from scratch...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    df = pd.read_csv(metadata_path)

    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    input_root = "./input"

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(input_root, row["image_path"])
        # Read as grayscale to ensure (H, W)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load Mask (if not test)
        if mode != "test":
            mask_path = os.path.join(input_root, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks_list.append(mask)

        images_list.append(img)
        depths_list.append(row["z"])
        ids_list.append(row["id"])

    # Convert to numpy arrays
    images_arr = np.array(images_list, dtype=np.uint8)
    depths_arr = np.array(depths_list, dtype=np.float32)
    ids_arr = np.array(ids_list)

    # Save cache
    np.save(cache_files["images"], images_arr)
    np.save(cache_files["depths"], depths_arr)
    np.save(cache_files["ids"], ids_arr)

    if mode != "test":
        masks_arr = np.array(masks_list, dtype=np.uint8)
        np.save(cache_files["masks"], masks_arr)
    else:
        masks_arr = None

    return images_arr, masks_arr, depths_arr, ids_arr


def get_dataloaders(
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    train_metadata="./metadata/train.csv",
    val_metadata="./metadata/val.csv",
    test_metadata="./metadata/test.csv",
):
    """
    Creates DataLoaders for train, val, and test sets.
    Calculates depth statistics from the training set for normalization.
    """

    # 1. Load Data (with caching)
    train_imgs, train_masks, train_depths, train_ids = load_and_process_data(
        train_metadata, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = load_and_process_data(
        val_metadata, "val", load_cached_data
    )
    test_imgs, _, test_depths, test_ids = load_and_process_data(
        test_metadata, "test", load_cached_data
    )

    # 2. Calculate Depth Statistics (from Train only)
    depth_mean = train_depths.mean()
    depth_std = train_depths.std()

    # 3. Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        depth_mean,
        depth_std,
        mode="train",
    )
    val_dataset = SaltDataset(
        val_imgs, val_masks, val_depths, val_ids, depth_mean, depth_std, mode="val"
    )
    test_dataset = SaltDataset(
        test_imgs, None, test_depths, test_ids, depth_mean, depth_std, mode="test"
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
