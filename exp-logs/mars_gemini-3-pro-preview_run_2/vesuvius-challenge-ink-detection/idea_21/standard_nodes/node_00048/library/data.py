import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def project_volume_slab(volume_chunk):
    """
    Projects a 3D volume chunk into a 3-channel 2D image using the
    'Overlapping Thick Slab' strategy.

    Args:
        volume_chunk (np.ndarray): 3D array of shape (Depth, Height, Width).
                                   Depth must be exactly 24.

    Returns:
        np.ndarray: Projected image of shape (Height, Width, 3).
    """
    if volume_chunk.shape[0] != 24:
        raise ValueError(f"Expected volume depth of 24, got {volume_chunk.shape[0]}")

    # Define slab indices based on Config
    # Channel 0: Slices [0:12]
    start_0 = 0
    end_0 = start_0 + Config.SLAB_THICKNESS

    # Channel 1: Slices [6:18] (Offset by SLAB_STRIDE)
    start_1 = Config.SLAB_STRIDE
    end_1 = start_1 + Config.SLAB_THICKNESS

    # Channel 2: Slices [12:24] (Offset by 2 * SLAB_STRIDE)
    start_2 = start_1 + Config.SLAB_STRIDE
    end_2 = start_2 + Config.SLAB_THICKNESS

    # Calculate Maximum Intensity Projection (MIP) for each slab
    # axis=0 is the depth dimension
    ch0 = np.max(volume_chunk[start_0:end_0], axis=0)
    ch1 = np.max(volume_chunk[start_1:end_1], axis=0)
    ch2 = np.max(volume_chunk[start_2:end_2], axis=0)

    # Stack to create (Height, Width, 3)
    projected = np.stack([ch0, ch1, ch2], axis=-1)

    return projected


def load_fragment_slab(fragment_id, z_range):
    """
    Loads the specific Z-slab for a full fragment into memory.

    Args:
        fragment_id (str): ID of the fragment (e.g., '1').
        z_range (tuple): (start_slice, end_slice).

    Returns:
        np.ndarray: 3D volume of shape (Depth, Height, Width).
    """
    start_z, end_z = z_range
    # Determine if it's train or test based on existence
    # The metadata logic usually handles paths, but here we construct path for raw volume loading
    # We check train first, then test

    base_path = os.path.join(Config.INPUT_DIR, "train", fragment_id, "surface_volume")
    if not os.path.exists(base_path):
        base_path = os.path.join(
            Config.INPUT_DIR, "test", fragment_id, "surface_volume"
        )

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Surface volume not found for fragment {fragment_id}")

    slices = []
    for z in range(start_z, end_z):
        filename = f"{z:02d}.tif"
        path = os.path.join(base_path, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Slice {path} not found.")

        # Load slice
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image {path}")
        slices.append(img)

    # Stack slices -> (Depth, H, W)
    volume = np.stack(slices, axis=0)
    return volume


def _process_dataset(metadata_df, specialist_key):
    """
    Internal function to process raw data into arrays based on metadata.
    Optimized to load full fragment slabs once rather than per-patch.

    Args:
        metadata_df (pd.DataFrame): Metadata containing patch info.
        specialist_key (str): 'A', 'B', or 'C'.

    Returns:
        dict: {'images': np.ndarray, 'labels': np.ndarray}
    """
    z_range = Config.SPECIALIST_RANGES[specialist_key]

    # Group by fragment to optimize loading
    fragment_ids = metadata_df["fragment_id"].unique()

    all_images = []
    all_labels = []

    print(f"Processing data for Specialist {specialist_key} (Z-range {z_range})...")

    for fid in fragment_ids:
        # Filter metadata for this fragment
        frag_meta = metadata_df[metadata_df["fragment_id"] == fid]

        # Load the full slab for this fragment
        print(f"  Loading volume slab for fragment {fid}...")
        full_volume = load_fragment_slab(str(fid), z_range)

        # Project the full volume once
        # Output shape: (H_frag, W_frag, 3)
        full_projection = project_volume_slab(full_volume)

        # Load full label mask
        # Note: Test set won't have inklabels, but this function is primarily for train/val generation
        label_path = os.path.join(Config.INPUT_DIR, "train", str(fid), "inklabels.png")
        if os.path.exists(label_path):
            full_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback for test or missing labels (should not happen for train/val)
            full_label = np.zeros(full_projection.shape[:2], dtype=np.uint8)

        # Iterate through patches
        for _, row in frag_meta.iterrows():
            x, y = row["x"], row["y"]
            w, h = row["width"], row["height"]

            # Crop image
            # The projection is HWC
            patch_img = full_projection[y : y + h, x : x + w, :]

            # Crop label
            patch_label = full_label[y : y + h, x : x + w]

            # Ensure shape consistency
            if (
                patch_img.shape[0] != Config.TILE_SIZE
                or patch_img.shape[1] != Config.TILE_SIZE
            ):
                continue

            all_images.append(patch_img)
            all_labels.append(patch_label)

    # Stack into arrays
    # Keeping as original dtype (likely uint16) to save space in cache
    images_arr = np.stack(all_images, axis=0)
    labels_arr = np.stack(all_labels, axis=0)

    return {"images": images_arr, "labels": labels_arr}


def load_or_generate_data(split, specialist_key, load_cached_data=True):
    """
    Loads processed data from cache or generates it from scratch.
    Strictly follows the caching logic requirement.

    Args:
        split (str): 'train' or 'validation'.
        specialist_key (str): 'A', 'B', or 'C'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {'images': np.ndarray, 'labels': np.ndarray}
    """
    cache_filename = f"data_{split}_{specialist_key}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(
                f"Loading cached {split} data for Specialist {specialist_key} from {cache_path}..."
            )
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Proceeding to regenerate.")

    # 2. IF loading fails OR load_cached_data is False:
    # Compute/process the data from scratch.
    csv_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file {csv_path} not found.")

    df = pd.read_csv(csv_path)
    data = _process_dataset(df, specialist_key)

    # Save the result to the cache directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Saving generated data to {cache_path}...")
    np.save(cache_path, data)

    # 3. Return the data.
    return data


class InkDataset(Dataset):
    """
    PyTorch Dataset for Vesuvius Ink Detection.
    Handles on-the-fly normalization and augmentation.
    """

    def __init__(self, data_dict, transforms=None):
        """
        Args:
            data_dict (dict): Dictionary containing 'images' and 'labels' arrays.
            transforms (albumentations.Compose): Augmentations to apply.
        """
        self.images = data_dict["images"]
        self.labels = data_dict["labels"]
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        # Image: (H, W, 3) - likely uint16
        # Label: (H, W) - uint8
        image = self.images[idx]
        label = self.labels[idx]

        # Normalize to [0, 1] float32
        # We assume 16-bit input data (0-65535)
        image = image.astype(np.float32) / 65535.0
        label = label.astype(np.float32)

        # Apply transforms (Augmentations)
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Ensure label has channel dimension (1, H, W)
        # Albumentations ToTensorV2 returns (H, W) for masks usually
        if isinstance(label, torch.Tensor):
            if len(label.shape) == 2:
                label = label.unsqueeze(0)
        elif isinstance(label, np.ndarray):
            if len(label.shape) == 2:
                label = np.expand_dims(label, axis=0)

        return image, label


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.

    Args:
        mode (str): 'train' or 'validation'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Geometric only; no intensity augmentations
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_specialist_datasets(specialist_key, load_cached_data=True):
    """
    Factory function to get train and validation datasets for a specific specialist.

    Args:
        specialist_key (str): 'A', 'B', or 'C'.
        load_cached_data (bool): Use caching.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Train
    train_data = load_or_generate_data("train", specialist_key, load_cached_data)
    train_dataset = InkDataset(train_data, transforms=get_transforms("train"))

    # Validation
    val_data = load_or_generate_data("validation", specialist_key, load_cached_data)
    val_dataset = InkDataset(val_data, transforms=get_transforms("validation"))

    return train_dataset, val_dataset
