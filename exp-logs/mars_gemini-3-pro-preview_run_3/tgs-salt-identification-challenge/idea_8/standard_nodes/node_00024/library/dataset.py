import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase="train"):
    """
    Returns the albumentations transform pipeline for the given phase.

    Args:
        phase (str): 'train' or 'valid'/'test'.
    """
    # Pad to 128x128 (closest power of 2 > 101) using reflection to maintain texture
    transforms = [
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        )
    ]

    if phase == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                # Conservative ShiftScaleRotate as per strategy
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                # Random brightness/contrast for robustness
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def load_data(df, cache_name, load_cached_data=True):
    """
    Loads data from the dataframe, utilizing caching to speed up subsequent runs.

    Args:
        df (pd.DataFrame): Dataframe containing metadata (paths, ids, depths).
        cache_name (str): Unique identifier for the cache (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, images, masks, depths)
               ids: np.array of strings
               images: np.array of shape (N, 101, 101)
               masks: np.array of shape (N, 101, 101) or None
               depths: np.array of shape (N,)
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    path_ids = os.path.join(Config.CACHE_DIR, f"{cache_name}_ids.npy")
    path_images = os.path.join(Config.CACHE_DIR, f"{cache_name}_images.npy")
    path_masks = os.path.join(Config.CACHE_DIR, f"{cache_name}_masks.npy")
    path_depths = os.path.join(Config.CACHE_DIR, f"{cache_name}_depths.npy")

    has_masks = "mask_path" in df.columns

    # Attempt to load from cache
    if load_cached_data:
        try:
            print(f"[{cache_name}] Attempting to load cached data...")
            ids = np.load(path_ids, allow_pickle=True)

            if len(ids) != len(df):
                raise ValueError(f"Cache size mismatch: {len(ids)} vs {len(df)}")

            images = np.load(path_images)
            depths = np.load(path_depths)

            if has_masks:
                if os.path.exists(path_masks):
                    masks = np.load(path_masks)
                    print(
                        f"[{cache_name}] Loaded {len(ids)} samples with masks from cache."
                    )
                    return ids, images, masks, depths
                else:
                    print(
                        f"[{cache_name}] Cache incomplete (missing masks). Reloading..."
                    )
            else:
                print(f"[{cache_name}] Loaded {len(ids)} samples from cache.")
                return ids, images, None, depths

        except (FileNotFoundError, IOError, ValueError) as e:
            print(
                f"[{cache_name}] Cache not found or invalid ({e}). Processing from scratch..."
            )

    # Process data from scratch
    print(f"[{cache_name}] Loading images from disk...")

    ids_list = df["id"].values
    depths_list = df["z"].values

    images_list = []
    masks_list = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images_list.append(img)

        # Load Mask if available
        if has_masks:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            # Ensure binary mask (0 or 1)
            mask = (mask > 127).astype(np.uint8)
            masks_list.append(mask)

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float32)

    # Save to cache
    np.save(path_ids, ids_list)
    np.save(path_images, images)
    np.save(path_depths, depths)

    if has_masks:
        masks = np.array(masks_list, dtype=np.uint8)
        np.save(path_masks, masks)
        print(
            f"[{cache_name}] Processed and cached {len(ids_list)} samples with masks."
        )
        return ids_list, images, masks, depths
    else:
        print(f"[{cache_name}] Processed and cached {len(ids_list)} samples.")
        return ids_list, images, None, depths


class SaltDataset(Dataset):
    def __init__(self, images, depths, masks=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            depths (np.ndarray): Array of depths (N,).
            masks (np.ndarray, optional): Array of masks (N, H, W).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve data
        image = self.images[idx]  # Shape: (101, 101), uint8
        depth = self.depths[idx]  # Scalar

        # 2. Prepare for Augmentation
        # Albumentations expects HWC or HW
        data = {"image": image}
        if self.masks is not None:
            data["mask"] = self.masks[idx]

        # 3. Apply Augmentations
        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if self.masks is not None:
                mask = augmented["mask"]

        # Note: ToTensorV2 converts image to float32 tensor and scales if specified,
        # but here we handle normalization manually to control channel multiplexing.
        # ToTensorV2 output is (C, H, W) if channels exist, or (H, W) if not.
        # Since input was (H, W), output of ToTensorV2 is (1, H, W) or (H, W).
        # We need to be careful. Albumentations usually preserves shape for grayscale unless specified.
        # Let's inspect: ToTensorV2 moves channel first.

        # If ToTensorV2 was the last step, 'image' is now a Tensor.
        # We need to construct the 3-channel input: [Seismic, Seismic, Depth]

        # Extract tensor content
        if isinstance(image, torch.Tensor):
            image = image.float()
            # If (1, H, W), squeeze. If (H, W), keep.
            if image.ndim == 3:
                image = image.squeeze(0)
        else:
            image = torch.from_numpy(image).float()

        # Normalize Image to [0, 1]
        image = image / 255.0

        # Normalize Depth to [0, 1] (Assuming max depth approx 1000 based on dataset info 51-959)
        depth_norm = float(depth) / 1000.0

        # 4. Construct 3-Channel Input
        # Shape: (3, H, W)
        h, w = image.shape
        depth_channel = torch.full((1, h, w), depth_norm, dtype=torch.float32)

        # Stack: [Seismic, Seismic, Depth]
        # image.unsqueeze(0) makes it (1, H, W)
        image_channel = image.unsqueeze(0)
        input_tensor = torch.cat([image_channel, image_channel, depth_channel], dim=0)

        # 5. Handle Mask
        if self.masks is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                mask = torch.from_numpy(mask).float()

            # Ensure mask is (H, W) or (1, H, W)
            if mask.ndim == 3:
                mask = mask.squeeze(0)

            return input_tensor, mask

        return input_tensor, torch.zeros((1,))  # Dummy target for test set
