import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import rle_decode


def get_depth_stats():
    """
    Calculates or retrieves cached depth statistics (mean, std) for standardization.
    """
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        return df["mean"].item(), df["std"].item()

    # Calculate from source if not cached
    if os.path.exists(Config.DEPTHS_CSV_PATH):
        df = pd.read_csv(Config.DEPTHS_CSV_PATH)
        mean_val = df["z"].mean()
        std_val = df["z"].std()

        # Cache results
        pd.DataFrame({"mean": [mean_val], "std": [std_val]}).to_csv(
            stats_path, index=False
        )
        return mean_val, std_val

    # Fallback
    return 0.0, 1.0


def get_transforms(mode="train"):
    """
    Constructs the Albumentations transform pipeline.
    """
    transforms = []

    # 1. Spatial Alignment: Pad to 128x128 with Reflection
    # We pad first to ensure consistent spatial dimensions for the model
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            always_apply=True,
        )
    )

    if mode == "train":
        # Non-Rigid Augmentation (Elastic)
        transforms.append(
            A.ElasticTransform(
                alpha=Config.AUG_ELASTIC_ALPHA,
                sigma=Config.AUG_ELASTIC_SIGMA,
                alpha_affine=None,
                p=Config.AUG_ELASTIC_P,
            )
        )

        # Rigid Augmentation (Geometric)
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=Config.AUG_RIGID_P,
            )
        )

        # Horizontal Flip
        transforms.append(A.HorizontalFlip(p=0.5))

    # Normalization
    # Handle 1-channel vs 3-channel config mismatch
    norm_mean = Config.NORM_MEAN
    norm_std = Config.NORM_STD
    if Config.CHANNELS == 1 and len(norm_mean) == 3:
        norm_mean = [np.mean(Config.NORM_MEAN)]
        norm_std = [np.mean(Config.NORM_STD)]

    transforms.append(
        A.Normalize(
            mean=norm_mean, std=norm_std, max_pixel_value=255.0, always_apply=True
        )
    )

    # Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def load_dataset_arrays(
    metadata_path, cache_prefix, load_cached_data=True, debug_size=None
):
    """
    Loads dataset arrays from disk or cache.
    Caches raw 101x101 images/masks to .npy files for fast loading.

    Args:
        metadata_path (str): Path to metadata CSV.
        cache_prefix (str): Unique prefix for cache files (e.g., 'train_fold0').
        load_cached_data (bool): If True, attempts to load from .npy cache.
        debug_size (int): Optional limit on number of samples.

    Returns:
        tuple: (images, masks, depths, ids)
    """
    # Define cache file paths
    p_images = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    p_masks = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_masks.npy")
    p_depths = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_depths.npy")
    p_ids = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    cache_exists = (
        os.path.exists(p_images)
        and os.path.exists(p_masks)
        and os.path.exists(p_depths)
        and os.path.exists(p_ids)
    )

    # 1. Try Loading from Cache
    if load_cached_data and cache_exists:
        images = np.load(p_images)
        masks = np.load(p_masks)
        depths = np.load(p_depths)
        ids = np.load(p_ids)

        if debug_size is not None:
            return (
                images[:debug_size],
                masks[:debug_size],
                depths[:debug_size],
                ids[:debug_size],
            )
        return images, masks, depths, ids

    # 2. Process from Scratch
    df = pd.read_csv(metadata_path)
    if debug_size is not None:
        df = df.iloc[:debug_size]

    n_samples = len(df)

    # Allocate arrays
    images = np.zeros((n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
    masks = np.zeros((n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
    depths = np.zeros((n_samples,), dtype=np.float32)
    ids = np.empty((n_samples,), dtype=object)

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        # Read as grayscale (H, W)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Safety fallback
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
        images[idx] = img

        # Load Mask
        # If rle_mask exists (Train/Val), decode it. Else (Test), leave as 0.
        if "rle_mask" in row and pd.notna(row["rle_mask"]):
            masks[idx] = rle_decode(
                row["rle_mask"], shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)
            )

        # Load Depth
        depths[idx] = row["z"]

        # Load ID
        ids[idx] = row["id"]

    # 3. Save to Cache (only if full dataset processed)
    if debug_size is None:
        np.save(p_images, images)
        np.save(p_masks, masks)
        np.save(p_depths, depths)
        np.save(p_ids, ids)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        ids,
        transforms=None,
        mode="train",
        soft_labels=None,
    ):
        """
        PyTorch Dataset for Salt Segmentation.

        Args:
            images (np.ndarray): Array of images (N, 101, 101).
            masks (np.ndarray): Array of binary masks (N, 101, 101).
            depths (np.ndarray): Array of depths (N,).
            ids (np.ndarray): Array of IDs (N,).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', 'test', 'pseudo'.
            soft_labels (np.ndarray, optional): Soft masks for distillation (N, 101, 101).
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms
        self.mode = mode
        self.soft_labels = soft_labels

        # Depth Standardization Stats
        self.depth_mean, self.depth_std = get_depth_stats()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Data
        img = self.images[idx]  # (101, 101)

        # Determine target mask
        if self.mode == "pseudo" and self.soft_labels is not None:
            mask = self.soft_labels[idx]  # Float soft mask
        else:
            mask = self.masks[idx]  # Binary mask

        # 2. Apply Transforms
        data = {"image": img}

        # Only pass mask to transforms if we are in a mode that uses it
        if self.mode in ["train", "val", "pseudo"]:
            data["mask"] = mask

        if self.transforms:
            augmented = self.transforms(**data)
            img_tensor = augmented["image"]

            if "mask" in data:
                mask_tensor = augmented["mask"]
                # Ensure mask is float for Loss functions (BCE/Lovasz)
                mask_tensor = mask_tensor.float()

                # Add channel dim if missing (Albumentations might return H,W or 1,H,W depending on config)
                # ToTensorV2 usually doesn't add channel to mask unless specified.
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
            else:
                # Placeholder for test mode
                mask_tensor = torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))
        else:
            # Fallback without transforms
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # 3. Process Depth
        # Standardize
        d_val = self.depths[idx]
        d_norm = (d_val - self.depth_mean) / self.depth_std
        depth_tensor = torch.tensor([d_norm], dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "depth": depth_tensor,
            "id": self.ids[idx],
        }
