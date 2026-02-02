import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_metadata, load_image


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.

    Features:
    - Loads image pairs (Noisy, Clean) based on metadata.
    - Pre-loads all images into RAM for efficient training (dataset is small).
    - Implements random cropping and geometric augmentations for the training split.
    - Provides full images for validation and testing.
    """

    def __init__(self, split, debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.split = split
        self.patch_size = Config.PATCH_SIZE

        # Load metadata dataframe
        self.metadata = get_metadata(split)

        # Optional debugging limit
        if debug:
            self.metadata = self.metadata.iloc[:10]

        # Pre-load images into memory to avoid disk I/O during training
        self.samples = self._load_images()

    def _load_images(self):
        """
        Internal method to load all images defined in the metadata into a list of dictionaries.

        Returns:
            list: List of dicts, e.g., [{'id': '101', 'noisy': np.ndarray, 'clean': np.ndarray}, ...]
        """
        samples = []
        input_dir = Config.INPUT_DIR

        for _, row in self.metadata.iterrows():
            item = {"id": str(row["id"])}

            # Construct paths
            noisy_path = os.path.join(input_dir, row["noisy_image_path"])

            # Load Noisy Image
            try:
                item["noisy"] = load_image(noisy_path)
            except FileNotFoundError:
                # Should be caught by metadata verification, but safe to handle
                continue

            # Load Clean Image (Only for train and val)
            if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
                clean_path = os.path.join(input_dir, row["clean_image_path"])
                try:
                    item["clean"] = load_image(clean_path)
                except FileNotFoundError:
                    continue

            samples.append(item)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple:
                - Train: (noisy_patch_tensor, clean_patch_tensor)
                - Val:   (noisy_full_tensor, clean_full_tensor)
                - Test:  (noisy_full_tensor, image_id_str)
        """
        sample = self.samples[idx]
        noisy = sample["noisy"]  # Shape: (H, W), dtype: float32, range: [0, 1]

        if self.split == "train":
            clean = sample["clean"]

            h, w = noisy.shape
            th, tw = self.patch_size, self.patch_size

            # --- 1. Random Crop ---
            # Ensure image dimensions are sufficient for cropping
            if h >= th and w >= tw:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)

                noisy_patch = noisy[i : i + th, j : j + tw]
                clean_patch = clean[i : i + th, j : j + tw]
            else:
                # Fallback for small images (unlikely based on dataset analysis): use full image
                # Note: This might cause batch collation errors if sizes vary,
                # but dataset analysis confirms min dims > 128.
                noisy_patch = noisy
                clean_patch = clean

            # --- 2. Geometric Augmentations ---
            # Apply same transforms to both noisy and clean to maintain alignment

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            # Random 90-degree Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # --- 3. To Tensor ---
            # np.ascontiguousarray is required because flip/rot can create negative strides
            # which torch.from_numpy does not support.
            noisy_t = (
                torch.from_numpy(np.ascontiguousarray(noisy_patch)).unsqueeze(0).float()
            )
            clean_t = (
                torch.from_numpy(np.ascontiguousarray(clean_patch)).unsqueeze(0).float()
            )

            return noisy_t, clean_t

        elif self.split == "val":
            clean = sample["clean"]
            # Return full images for validation
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
            clean_t = torch.from_numpy(clean).unsqueeze(0).float()
            return noisy_t, clean_t

        elif self.split == "test":
            # Return full image and ID for inference
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
            return noisy_t, sample["id"]
