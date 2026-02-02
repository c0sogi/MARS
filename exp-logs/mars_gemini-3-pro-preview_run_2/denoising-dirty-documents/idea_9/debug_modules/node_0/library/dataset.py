import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import read_image, normalize, set_seed


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising task.
    Handles loading, caching, and augmenting image data.
    Implements High-Density Sampling for training.
    """

    def __init__(self, subset: str, config: Config, load_cached_data: bool = True):
        """
        Args:
            subset (str): One of 'train', 'val', 'test'.
            config (Config): Configuration object.
            load_cached_data (bool): Whether to load data from cache if available.
        """
        super().__init__()
        self.subset = subset
        self.config = config
        self.patches_per_image = config.PATCHES_PER_IMAGE if subset == "train" else 1

        # Set seed for reproducibility
        set_seed(config.SEED)

        # Define Metadata Path
        if subset == "train":
            self.metadata_path = config.TRAIN_METADATA
        elif subset == "val":
            self.metadata_path = config.VAL_METADATA
        elif subset == "test":
            self.metadata_path = config.TEST_METADATA
        else:
            raise ValueError(f"Invalid subset: {subset}")

        # Cache Directory
        self.cache_dir = os.path.join(config.WORKING_DIR, "cache", subset)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Data
        self.data = self._load_data(load_cached_data)

        # Define Augmentations
        if subset == "train":
            self.transform = A.Compose(
                [
                    A.RandomCrop(height=config.PATCH_SIZE, width=config.PATCH_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ],
                additional_targets={"label": "image"},
            )
        else:
            self.transform = A.Compose(
                [ToTensorV2()], additional_targets={"label": "image"}
            )

    def _load_data(self, load_cached_data: bool):
        """
        Loads data from metadata CSV.
        Implements caching logic: Check Cache -> Load or Compute -> Save Cache.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)
        loaded_data = []

        for _, row in df.iterrows():
            img_id = str(row["id"])

            # Paths for cache
            noisy_cache_path = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
            clean_cache_path = (
                os.path.join(self.cache_dir, f"{img_id}_clean.npy")
                if self.subset != "test"
                else None
            )

            # 1. Try to load from cache
            noisy_img = None
            clean_img = None

            if load_cached_data:
                if os.path.exists(noisy_cache_path):
                    try:
                        noisy_img = np.load(noisy_cache_path)
                    except Exception:
                        noisy_img = None

                if clean_cache_path and os.path.exists(clean_cache_path):
                    try:
                        clean_img = np.load(clean_cache_path)
                    except Exception:
                        clean_img = None

            # 2. If loading failed or not requested, compute from scratch
            if noisy_img is None:
                feature_rel_path = row["feature_path"]
                full_feature_path = os.path.join(
                    self.config.INPUT_DIR, feature_rel_path
                )
                # Read and normalize
                raw_noisy = read_image(full_feature_path, grayscale=True)
                noisy_img = normalize(raw_noisy)
                # Save to cache
                np.save(noisy_cache_path, noisy_img)

            if self.subset != "test" and clean_img is None:
                label_rel_path = row["label_path"]
                full_label_path = os.path.join(self.config.INPUT_DIR, label_rel_path)
                # Read and normalize
                raw_clean = read_image(full_label_path, grayscale=True)
                clean_img = normalize(raw_clean)
                # Save to cache
                np.save(clean_cache_path, clean_img)

            # Store in memory
            item = {
                "id": img_id,
                "noisy": noisy_img,  # Shape: (H, W), float32 [0, 1]
            }
            if clean_img is not None:
                item["clean"] = clean_img

            loaded_data.append(item)

        return loaded_data

    def __len__(self):
        """
        Returns the total number of samples.
        For training, this is num_images * patches_per_image (High-Density Sampling).
        For val/test, this is num_images.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        """
        # Map global index to image index
        img_idx = idx // self.patches_per_image
        sample_data = self.data[img_idx]

        noisy = sample_data["noisy"]
        img_id = sample_data["id"]

        # Prepare for Albumentations (needs H, W, C or H, W)
        # Albumentations expects numpy arrays.

        if self.subset == "test":
            # Inference Mode
            # Just convert to tensor
            transformed = self.transform(image=noisy)
            noisy_tensor = transformed["image"]  # (C, H, W)
            return noisy_tensor, img_id

        else:
            # Train or Val
            clean = sample_data["clean"]

            # Apply transforms
            # We pass clean image as 'label' target to ensure same geometric transforms are applied
            transformed = self.transform(image=noisy, label=clean)

            noisy_tensor = transformed["image"]
            clean_tensor = transformed["label"]

            if self.subset == "train":
                # Return tuple (input, target)
                return noisy_tensor, clean_tensor
            else:
                # Validation: Return (input, target, id)
                return noisy_tensor, clean_tensor, img_id
