import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import set_seed

# ==========================================
# Constants & Physics Parameters
# ==========================================
# Ash Color Scheme Bounds (Brightness Temperatures in Kelvin)
# Derived from standard GOES-16 ABI recipes for Contrail/Ash detection
BOUNDS_T11 = (243, 303)  # Blue Channel
BOUNDS_T14_T11 = (-4, 5)  # Green Channel (Difference)
BOUNDS_T15_T13 = (-4, 2)  # Red Channel (Difference)


def normalize_range(data, bounds):
    """Normalizes data to [0, 1] based on provided min/max bounds."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


# ==========================================
# Dataset Class
# ==========================================
class ContrailsDataset(Dataset):
    def __init__(self, images, masks=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, 6).
            masks (np.ndarray, optional): Array of shape (N, H, W, 1).
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx]  # (H, W, 6)

        if self.masks is not None:
            mask = self.masks[idx]  # (H, W, 1)

            # Apply Augmentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask is channel-first (C, H, W)
            # Albumentations ToTensorV2 usually keeps mask as (H, W) or (H, W, 1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)

            return image, mask.float()

        else:
            # Test mode
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

            # Return dummy mask for consistency in loops
            return image, torch.zeros((1, image.shape[1], image.shape[2]))


# ==========================================
# Transforms
# ==========================================
def get_transforms(stage="train"):
    """
    Returns the Albumentations transform pipeline.
    Explicitly excludes elastic/grid distortions to preserve linearity.
    """
    if stage == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),  # Enables 90/180/270 rotations
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([ToTensorV2()])


# ==========================================
# Data Processing & Caching
# ==========================================
def process_record(record, input_dir, is_test=False):
    """
    Reads raw NPY bands and constructs the 6-channel input tensor.
    Channels 1-3: Ash False Color (Normalized)
    Channels 4-6: Raw Temporal Differences
    """

    # Helper to load specific band
    def load_band(b):
        # Path is relative: e.g., "train/id/band_11.npy"
        path = os.path.join(input_dir, record[f"band_{b:02d}"])
        return np.load(path)

    # Load required bands
    b11 = load_band(11)
    b13 = load_band(13)
    b14 = load_band(14)
    b15 = load_band(15)

    # Temporal indices
    # Sequence length is 8. n_times_before=4 -> Index 4 is current frame.
    t_curr = Config.N_TIMES_BEFORE
    t_prev = t_curr - 1

    # --- 1. Ash False Color Composite (t=4) ---
    # Red: T15 - T13
    r = normalize_range(b15[..., t_curr] - b13[..., t_curr], BOUNDS_T15_T13)
    # Green: T14 - T11
    g = normalize_range(b14[..., t_curr] - b11[..., t_curr], BOUNDS_T14_T11)
    # Blue: T11
    b = normalize_range(b11[..., t_curr], BOUNDS_T11)

    ash = np.stack([r, g, b], axis=-1)
    ash = np.clip(ash, 0, 1)

    # --- 2. Temporal Differences (t=4 - t=3) ---
    # Using Bands 11, 14, 15 as per Idea
    d11 = b11[..., t_curr] - b11[..., t_prev]
    d14 = b14[..., t_curr] - b14[..., t_prev]
    d15 = b15[..., t_curr] - b15[..., t_prev]

    diff = np.stack([d11, d14, d15], axis=-1)
    # Note: We do not clip differences to preserve dynamic range of cooling

    # Combine
    image = np.concatenate([ash, diff], axis=-1)  # (256, 256, 6)

    # Load Mask
    mask = None
    if not is_test:
        mask_path = os.path.join(input_dir, record["human_pixel_masks"])
        mask = np.load(mask_path)  # (256, 256, 1)

    return image, mask


def load_data(split="train", load_cached_data=True, sample_size=None):
    """
    Loads dataset from cache or processes from scratch.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_img_path = os.path.join(Config.CACHE_DIR, f"{split}_images.npy")
    cache_msk_path = os.path.join(Config.CACHE_DIR, f"{split}_masks.npy")

    # Check if cache exists
    has_cache = os.path.exists(cache_img_path)
    if split != "test":
        has_cache = has_cache and os.path.exists(cache_msk_path)

    # 1. Load from Cache
    if load_cached_data and has_cache:
        print(f"Loading {split} data from cache: {Config.CACHE_DIR}")
        images = np.load(cache_img_path)
        masks = np.load(cache_msk_path) if split != "test" else None

        # Handle subsetting for debugging
        if sample_size is not None:
            images = images[:sample_size]
            if masks is not None:
                masks = masks[:sample_size]

        return images, masks

    # 2. Process from Scratch
    print(f"Processing {split} data from scratch (Cache miss or force reload)...")

    # Select metadata file
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "validation":
        meta_path = Config.VALIDATION_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df = pd.read_csv(meta_path)

    # Subset if requested (before processing to save time)
    if sample_size is not None:
        df = df.head(sample_size)

    # Pre-allocate arrays
    n_samples = len(df)
    h, w = Config.IMAGE_SIZE, Config.IMAGE_SIZE
    c = Config.IN_CHANNELS

    all_images = np.zeros((n_samples, h, w, c), dtype=np.float32)
    all_masks = (
        np.zeros((n_samples, h, w, 1), dtype=np.uint8) if split != "test" else None
    )

    for idx, row in df.iterrows():
        # idx in df might not be 0..N if sampled, but iterrows returns index
        # We use a counter or reset index.
        # Since we use head(), indices are 0..N-1.
        # If we used sample(), we'd need enumerate.
        pass

    # Use enumerate to safely fill arrays
    for i, (_, row) in enumerate(df.iterrows()):
        img, msk = process_record(row, Config.INPUT_DIR, is_test=(split == "test"))
        all_images[i] = img.astype(np.float32)
        if split != "test":
            all_masks[i] = msk.astype(np.uint8)

    # Save to cache ONLY if we processed the full dataset
    if sample_size is None:
        print(f"Saving {split} data to cache...")
        np.save(cache_img_path, all_images)
        if split != "test":
            np.save(cache_msk_path, all_masks)

    return all_images, all_masks


def get_dataloader(
    split="train",
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    sample_size=None,
):
    """
    Factory function to create a DataLoader.
    """
    # Load data arrays
    images, masks = load_data(split, load_cached_data, sample_size)

    # Define transforms
    transform = get_transforms(stage=split if split == "train" else "valid")

    # Create Dataset
    dataset = ContrailsDataset(images, masks, transform=transform)

    # Create DataLoader
    shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # Drop last incomplete batch in training
    )

    return loader
