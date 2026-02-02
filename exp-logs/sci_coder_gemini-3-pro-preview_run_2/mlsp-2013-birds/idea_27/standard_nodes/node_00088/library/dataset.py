import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


def load_images(df, cache_name, load_cached_data=True):
    """
    Loads images based on the provided dataframe.
    Implements deterministic caching of pre-processed (resized, pseudo-RGB) images.

    Args:
        df (pd.DataFrame): Dataframe containing file information.
        cache_name (str): Unique identifier for the cache file (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of images with shape (N, H, W, 3).
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"images_{cache_name}.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            images = np.load(cache_path)
            # Validate cache consistency with current dataframe
            if len(images) == len(df):
                print(f"Loaded {len(images)} images from cache: {cache_path}")
                return images
            else:
                print(
                    f"Cache size mismatch ({len(images)} vs {len(df)}). Reloading from source."
                )
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    print(f"Processing {len(df)} images for {cache_name}...")
    images = []

    # Iterate through dataframe to load and process images
    for _, row in df.iterrows():
        # Metadata contains paths like 'supplemental_data/spectrograms/PC10_....bmp'
        # We enforce usage of Filtered Spectrograms defined in Config
        filename = os.path.basename(row["file_path_spec"])
        full_path = os.path.join(Config.SPECTROGRAM_DIR, filename)

        # Load image (BMPs are typically single channel)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing files (should be rare given metadata checks)
            # Create a black image of target size
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
        else:
            # Resize to target dimensions (Width, Height)
            # Config.IMG_WIDTH is Time (448), Config.IMG_HEIGHT is Frequency (224)
            img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))

        # Convert to Pseudo-RGB (3 channels) for ImageNet compatibility
        img = cv2.merge([img, img, img])

        images.append(img)

    images = np.array(images, dtype=np.uint8)

    # Save to cache for future runs
    try:
        np.save(cache_path, images)
        print(f"Saved processed images to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return images


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Supports Hard Labels (Multi-hot) and Soft Labels (for Distillation).
    """

    def __init__(self, images, df, transforms=None, soft_labels=None):
        """
        Args:
            images (np.ndarray): Pre-loaded images (N, H, W, 3).
            df (pd.DataFrame): Dataframe containing metadata and labels.
            transforms (A.Compose, optional): Albumentations transforms.
            soft_labels (np.ndarray, optional): Soft targets for distillation (N, Num_Classes).
        """
        self.images = images
        self.rec_ids = df["rec_id"].values
        self.transforms = transforms

        # Extract Hard Labels (Multi-hot encoded)
        # Look for columns starting with 'species_'
        label_cols = [c for c in df.columns if c.startswith("species_")]
        if label_cols:
            self.labels = df[label_cols].values.astype(np.float32)
        else:
            # If no labels present (e.g., pure inference), initialize zeros
            self.labels = np.zeros((len(df), Config.NUM_CLASSES), dtype=np.float32)

        # Store Soft Labels if provided (Generation 1 / Distillation phase)
        if soft_labels is not None:
            self.soft_labels = soft_labels.astype(np.float32)
        else:
            self.soft_labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        rec_id = self.rec_ids[idx]
        target = self.labels[idx]

        # Apply Augmentations / Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform: Normalize to [0, 1] and convert to Tensor
            # Assumes image is HWC, converts to CHW
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare Soft Targets
        if self.soft_labels is not None:
            soft_target = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        else:
            # Return zeros if no soft labels are available
            # This ensures consistent return signature for the collate function
            soft_target = torch.zeros_like(torch.tensor(target))

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "soft_target": soft_target,
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }
