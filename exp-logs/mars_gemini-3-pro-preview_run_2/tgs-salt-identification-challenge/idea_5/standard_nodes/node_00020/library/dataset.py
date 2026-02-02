import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_depth_stats():
    """
    Calculates mean and standard deviation for depths from the global depths file.

    Returns:
        tuple: (mean, std) of depth values.
    """
    df = pd.read_csv(Config.DEPTHS_CSV)
    depths = df["z"].values.astype(np.float32)
    return depths.mean(), depths.std()


def load_data(csv_path, cache_prefix, load_cached_data=True):
    """
    Loads data from CSV and images/masks, with caching mechanism.

    Args:
        csv_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for cached files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'images', 'masks', 'depths', 'ids'.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{cache_prefix}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{cache_prefix}_depths.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(depth_cache_path)
            and os.path.exists(id_cache_path)
        ):

            print(f"Loading cached data for {cache_prefix}...")
            images = np.load(img_cache_path)
            depths = np.load(depth_cache_path)
            ids = np.load(id_cache_path, allow_pickle=True)

            masks = None
            if os.path.exists(mask_cache_path):
                masks = np.load(mask_cache_path)

            return {"images": images, "masks": masks, "depths": depths, "ids": ids}

    # Process from scratch
    print(f"Processing data for {cache_prefix} from scratch...")
    df = pd.read_csv(csv_path)

    ids = df["id"].values
    depths = df["z"].values.astype(np.float32)

    # Pre-allocate arrays
    n_samples = len(df)
    h, w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE
    images = np.zeros((n_samples, h, w), dtype=np.uint8)

    has_masks = "mask_path" in df.columns
    masks = np.zeros((n_samples, h, w), dtype=np.uint8) if has_masks else None

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images[idx] = img

        # Load Mask
        if has_masks:
            mask_path = os.path.join(Config.INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            # Binarize
            masks[idx] = (mask > 127).astype(np.uint8)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)
    if has_masks:
        np.save(mask_cache_path, masks)

    return {"images": images, "masks": masks, "depths": depths, "ids": ids}


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    transforms = []

    # 1. Padding (101 -> 128)
    # Use reflection padding to handle boundary effects
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    )

    if phase == "train":
        # 2. Rigid Augmentations
        transforms.append(A.HorizontalFlip(p=Config.AUG_PROB))

        # ShiftScaleRotate
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=Config.SHIFT_LIMIT,
                scale_limit=Config.SCALE_LIMIT,
                rotate_limit=Config.ROTATE_LIMIT,
                border_mode=cv2.BORDER_REFLECT_101,
                p=Config.AUG_PROB,
            )
        )

        # 3. Non-Rigid Augmentations (Elastic)
        transforms.append(
            A.ElasticTransform(
                alpha=Config.ELASTIC_ALPHA,
                sigma=Config.ELASTIC_SIGMA,
                alpha_affine=Config.ELASTIC_ALPHA_AFFINE,
                border_mode=cv2.BORDER_REFLECT_101,
                p=Config.ELASTIC_PROB,
            )
        )

    # 4. Tensor Conversion
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    def __init__(self, data_dict, transform=None, depth_stats=None):
        """
        Args:
            data_dict (dict): Dictionary with 'images', 'masks', 'depths', 'ids'.
            transform (albumentations.Compose): Transforms to apply.
            depth_stats (tuple): (mean, std) for depth normalization.
        """
        self.images = data_dict["images"]
        self.masks = data_dict["masks"]
        self.depths = data_dict["depths"]
        self.ids = data_dict["ids"]
        self.transform = transform

        if depth_stats:
            self.depth_mean, self.depth_std = depth_stats
        else:
            self.depth_mean = self.depths.mean()
            self.depth_std = self.depths.std() + 1e-8

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx]  # (101, 101) uint8
        depth = self.depths[idx]  # float
        img_id = self.ids[idx]

        # Normalize depth (Standard Scaling)
        depth = (depth - self.depth_mean) / (self.depth_std + 1e-8)
        depth = torch.tensor([depth], dtype=torch.float32)

        # Apply Transforms
        if self.masks is not None:
            mask = self.masks[idx]  # (101, 101) uint8

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                T = ToTensorV2()
                augmented = T(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure channel dimension (C, H, W)
            # ToTensorV2 on (H, W) input returns (H, W) tensor usually, or (1, H, W)?
            # It preserves shape for 2D. We need to unsqueeze.
            if image.ndim == 2:
                image = image.unsqueeze(0)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # Normalize Image (0-1)
            image = image.float() / 255.0
            mask = mask.float()  # Binary 0.0 or 1.0

            return image, mask, depth, img_id

        else:
            # Test mode (No mask)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                T = ToTensorV2()
                augmented = T(image=image)
                image = augmented["image"]

            if image.ndim == 2:
                image = image.unsqueeze(0)

            image = image.float() / 255.0

            return image, depth, img_id
