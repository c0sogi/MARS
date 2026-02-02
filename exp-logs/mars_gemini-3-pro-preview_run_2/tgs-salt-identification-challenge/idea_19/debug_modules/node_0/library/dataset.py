import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import do_pad, rle_decode

# Constants
CACHE_DIR = "./working/idea_19"
INPUT_ROOT = "./input"
METADATA_ROOT = "./metadata"
IMG_SIZE_ORIG = 101
IMG_SIZE_PAD = 128

# ImageNet Stats for 1-channel (approx average of RGB)
# Mean: (0.485 + 0.456 + 0.406) / 3 = 0.449
# Std: (0.229 + 0.224 + 0.225) / 3 = 0.226
IMG_MEAN = 0.449
IMG_STD = 0.226


def get_depth_stats():
    """
    Calculates depth mean and std from the training metadata.
    Used to normalize depth across all splits consistently.
    """
    path = os.path.join(METADATA_ROOT, "train.csv")
    if not os.path.exists(path):
        # Fallback if metadata not generated yet
        return 0.0, 1.0
    df = pd.read_csv(path)
    z = df["z"].values
    return z.mean(), z.std()


def get_transforms(mode="train"):
    """
    Returns Albumentations transform pipeline.
    """
    if mode == "train":
        # Weak Augmentation
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    elif mode == "student":
        # Strong Augmentation for Noisy Student
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.CoarseDropout(max_holes=8, max_height=10, max_width=10, p=0.2),
                A.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode. Handles caching to disk.

    Args:
        mode: 'train', 'val', 'test'
        load_cached_data: If True, tries to load from numpy cache.

    Returns:
        dict: {'images': ..., 'masks': ..., 'depths': ..., 'ids': ...}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "images": os.path.join(CACHE_DIR, f"{mode}_images.npy"),
        "masks": os.path.join(CACHE_DIR, f"{mode}_masks.npy"),
        "depths": os.path.join(CACHE_DIR, f"{mode}_depths.npy"),
        "ids": os.path.join(CACHE_DIR, f"{mode}_ids.npy"),
    }

    # Check if all required cache files exist
    required_keys = ["images", "depths", "ids"]
    if mode != "test":
        required_keys.append("masks")

    cache_exists = all(os.path.exists(cache_files[k]) for k in required_keys)

    if load_cached_data and cache_exists:
        print(f"Loading {mode} data from cache...")
        data = {}
        data["images"] = np.load(cache_files["images"])
        data["depths"] = np.load(cache_files["depths"])
        data["ids"] = np.load(cache_files["ids"], allow_pickle=True)
        if mode != "test":
            data["masks"] = np.load(cache_files["masks"])
        else:
            data["masks"] = None
        return data

    print(f"Processing {mode} data from scratch...")

    # Load metadata
    meta_path = os.path.join(METADATA_ROOT, f"{mode}.csv")
    df = pd.read_csv(meta_path)

    images = []
    masks = []
    depths = []
    ids = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_ROOT, row["image_path"])
        # Load as grayscale (1 channel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

        # Load Mask (if available)
        if mode != "test":
            rle = row["rle_mask"]
            if pd.isna(rle) or rle == "":
                mask = np.zeros((IMG_SIZE_ORIG, IMG_SIZE_ORIG), dtype=np.uint8)
            else:
                mask = rle_decode(rle, shape=(IMG_SIZE_ORIG, IMG_SIZE_ORIG))
            masks.append(mask)

        # Load Depth
        depths.append(row["z"])

        # Load ID
        ids.append(row["id"])

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    if mode != "test":
        masks = np.array(masks, dtype=np.uint8)
        np.save(cache_files["masks"], masks)

    # Save to cache
    np.save(cache_files["images"], images)
    np.save(cache_files["depths"], depths)
    np.save(cache_files["ids"], ids)

    data = {
        "images": images,
        "depths": depths,
        "ids": ids,
        "masks": masks if mode != "test" else None,
    }
    return data


class SaltDataset(Dataset):
    def __init__(
        self,
        mode="train",
        load_cached_data=True,
        transform=None,
        depth_mask_prob=0.0,
        force_zero_depth=False,
        pseudo_labels=None,
    ):
        """
        Args:
            mode: 'train', 'val', 'test'
            load_cached_data: Use cached numpy arrays
            transform: Albumentations transform
            depth_mask_prob: Probability to set depth to 0 (Bernoulli masking)
            force_zero_depth: If True, always set depth to 0 (for inference)
            pseudo_labels: Optional dict {id: mask} or array (N, H, W). Overrides GT masks.
        """
        self.mode = mode
        self.transform = transform
        self.depth_mask_prob = depth_mask_prob
        self.force_zero_depth = force_zero_depth

        # Load Data
        data = load_data(mode, load_cached_data)
        self.images = data["images"]
        self.depths = data["depths"]
        self.ids = data["ids"]
        self.masks = data["masks"]

        self.pseudo_labels = pseudo_labels

        # Global Depth Stats
        self.depth_mean, self.depth_std = get_depth_stats()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Image (H, W)
        image = self.images[idx]
        img_id = self.ids[idx]

        # 2. Pad Image (H, W) -> (128, 128)
        image_padded = do_pad(image, pad_to=IMG_SIZE_PAD)

        # 3. Handle Mask
        mask_padded = None

        # Check for pseudo labels first
        if self.pseudo_labels is not None:
            if isinstance(self.pseudo_labels, dict):
                mask = self.pseudo_labels.get(img_id)
            else:
                mask = self.pseudo_labels[idx]

            if mask is not None:
                mask_padded = do_pad(mask, pad_to=IMG_SIZE_PAD)

        elif self.masks is not None:
            mask = self.masks[idx]
            mask_padded = do_pad(mask, pad_to=IMG_SIZE_PAD)

        # 4. Prepare for Albumentations (H, W, C)
        if image_padded.ndim == 2:
            image_padded = image_padded[..., np.newaxis]

        # 5. Apply Transforms
        if self.transform:
            if mask_padded is not None:
                augmented = self.transform(image=image_padded, mask=mask_padded)
                image_tensor = augmented["image"]
                mask_tensor = augmented["mask"]
            else:
                augmented = self.transform(image=image_padded)
                image_tensor = augmented["image"]
                mask_tensor = None
        else:
            # Fallback
            t = ToTensorV2()
            if mask_padded is not None:
                augmented = t(image=image_padded, mask=mask_padded)
                image_tensor = augmented["image"]
                mask_tensor = augmented["mask"]
            else:
                augmented = t(image=image_padded)
                image_tensor = augmented["image"]
                mask_tensor = None

        # 6. Depth Processing
        depth_val = self.depths[idx]
        z = (depth_val - self.depth_mean) / self.depth_std

        if self.force_zero_depth:
            z = 0.0
        elif self.depth_mask_prob > 0:
            if np.random.random() < self.depth_mask_prob:
                z = 0.0

        depth_tensor = torch.tensor([z], dtype=torch.float32)

        # 7. Return
        if mask_tensor is not None:
            # Ensure mask has channel dim (1, H, W)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            return image_tensor, mask_tensor.float(), depth_tensor, img_id
        else:
            return image_tensor, depth_tensor, img_id
