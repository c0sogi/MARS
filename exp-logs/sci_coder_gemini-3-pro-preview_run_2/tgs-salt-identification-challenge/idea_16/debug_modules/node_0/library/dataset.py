import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# Constants
CACHE_DIR = "./working/idea_16/"
INPUT_ROOT = "./input"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128

# ImageNet Mean/Std averaged for grayscale (R+G+B)/3
# Mean: (0.485 + 0.456 + 0.406) / 3 = 0.449
# Std: (0.229 + 0.224 + 0.225) / 3 = 0.226
IMAGENET_MEAN_GRAY = [0.449]
IMAGENET_STD_GRAY = [0.226]


def get_depth_stats(metadata_path="./metadata/train.csv"):
    """
    Calculates mean and std of depth from the training metadata.
    Used to standardize depth across Train/Val/Test.
    """
    if not os.path.exists(metadata_path):
        # Fallback or error if metadata is missing, though prompt guarantees existence.
        return 0.0, 1.0

    df = pd.read_csv(metadata_path)
    depths = df["z"].values.astype(np.float32)
    return np.mean(depths), np.std(depths)


def get_transforms(phase):
    """
    Returns Albumentations transform pipeline for the specified phase.
    """
    transforms = []

    # 1. Spatial Alignment: Pad to 128x128 using Reflection
    transforms.append(
        A.PadIfNeeded(
            min_height=IMG_SIZE_TARGET,
            min_width=IMG_SIZE_TARGET,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    )

    if phase == "train":
        # 2. Augmentations
        # Non-Rigid: Elastic Transform
        transforms.append(
            A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2)
        )
        # Rigid: ShiftScaleRotate
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            )
        )
        # Flip
        transforms.append(A.HorizontalFlip(p=0.5))

    # 3. Normalization & Tensor Conversion
    transforms.append(A.Normalize(mean=IMAGENET_MEAN_GRAY, std=IMAGENET_STD_GRAY))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    def __init__(
        self,
        mode,
        metadata_file,
        depth_mean,
        depth_std,
        root_dir=INPUT_ROOT,
        cache_dir=CACHE_DIR,
        load_cached_data=True,
        transform=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            metadata_file (str): Path to the metadata CSV.
            depth_mean (float): Mean depth from training set.
            depth_std (float): Std dev of depth from training set.
            root_dir (str): Root directory of input data.
            cache_dir (str): Directory to store/load cached numpy arrays.
            load_cached_data (bool): Whether to use caching.
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.transform = transform if transform else get_transforms(mode)
        self.depth_mean = depth_mean
        self.depth_std = depth_std

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load data (from cache or raw)
        self.ids, self.images, self.masks, self.depths = self._load_data(
            metadata_file, load_cached_data
        )

    def _load_data(self, metadata_file, load_cached_data):
        """
        Handles caching logic: Load from .npy if available and requested,
        otherwise process from scratch and save.
        """
        # Define cache paths
        cache_prefix = f"{self.mode}"
        path_ids = os.path.join(self.cache_dir, f"{cache_prefix}_ids.npy")
        path_imgs = os.path.join(self.cache_dir, f"{cache_prefix}_images.npy")
        path_masks = os.path.join(self.cache_dir, f"{cache_prefix}_masks.npy")
        path_depths = os.path.join(self.cache_dir, f"{cache_prefix}_depths.npy")

        # Check if cache exists
        cache_exists = (
            os.path.exists(path_ids)
            and os.path.exists(path_imgs)
            and os.path.exists(path_depths)
            and (self.mode == "test" or os.path.exists(path_masks))
        )

        if load_cached_data and cache_exists:
            # Load from cache
            ids = np.load(path_ids, allow_pickle=True)
            images = np.load(path_imgs)
            depths = np.load(path_depths)
            if self.mode != "test":
                masks = np.load(path_masks)
            else:
                masks = None
            return ids, images, masks, depths

        # Process from scratch
        df = pd.read_csv(metadata_file)

        ids_list = []
        images_list = []
        masks_list = []
        depths_list = []

        for _, row in df.iterrows():
            img_id = row["id"]
            # Paths in metadata are relative to input root (e.g., "train/images/xxxx.png")
            img_path = os.path.join(self.root_dir, row["image_path"])

            # Load Image (Grayscale)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            images_list.append(img)
            ids_list.append(img_id)
            depths_list.append(row["z"])

            if self.mode != "test":
                mask_path = os.path.join(self.root_dir, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # If mask is missing but listed, assume empty or raise error.
                    # Given dataset integrity checks, we assume it exists.
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                # Binarize mask (0 or 255 -> 0 or 1)
                mask = (mask > 127).astype(np.uint8)
                masks_list.append(mask)

        # Convert to numpy arrays
        ids = np.array(ids_list)
        images = np.array(images_list, dtype=np.uint8)  # (N, H, W)
        depths = np.array(depths_list, dtype=np.float32)

        if self.mode != "test":
            masks = np.array(masks_list, dtype=np.uint8)  # (N, H, W)
        else:
            masks = None

        # Save to cache
        np.save(path_ids, ids)
        np.save(path_imgs, images)
        np.save(path_depths, depths)
        if masks is not None:
            np.save(path_masks, masks)

        return ids, images, masks, depths

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get data
        image = self.images[idx]  # (H, W)
        depth_raw = self.depths[idx]
        img_id = self.ids[idx]

        # Standardize depth
        depth_norm = (depth_raw - self.depth_mean) / self.depth_std

        # Bernoulli Depth Masking (Train only)
        # With p=0.5, replace depth with dataset mean (0 after standardization)
        if self.mode == "train":
            if random.random() < 0.5:
                depth_norm = 0.0

        # Apply Transforms
        if self.mode != "test":
            mask = self.masks[idx]  # (H, W)
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].float()  # BCE prefers float

            # Mask needs to be (1, H, W) for some losses or (H, W) for others.
            # Albumentations ToTensorV2 doesn't add channel dim to mask by default if input is 2D.
            # We ensure it matches model expectation (usually N, H, W for Lovasz, N, 1, H, W for BCE).
            # The CombinedLoss handles dimensions, but let's keep it (H, W) here.

            return (
                image_tensor,
                mask_tensor,
                torch.tensor(depth_norm, dtype=torch.float32),
                img_id,
            )
        else:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
            # For test, we might want to force depth to 0 if implementing the "Generalist" inference strategy
            # strictly, but the prompt says "Use fixed depth 0" in the idea description for Stage 2 inference.
            # The dataset class should be flexible. If the user wants fixed depth 0, they can pass
            # depth_mean=0, depth_std=1 and raw depth 0, or handle it in the inference loop.
            # However, the dataset just returns the standardized depth.

            return image_tensor, torch.tensor(depth_norm, dtype=torch.float32), img_id
