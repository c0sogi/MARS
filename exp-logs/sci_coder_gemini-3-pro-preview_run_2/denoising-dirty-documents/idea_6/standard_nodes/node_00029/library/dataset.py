import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset class for the denoising task.

    Features:
    - Caches preprocessed (normalized) images as .npy files.
    - Implements High-Density Sampling for training (random patches).
    - Provides full images for validation and testing.
    - Applies geometric augmentations during training.
    """

    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
        """
        self.mode = mode
        self.patch_size = Config.PATCH_SIZE
        self.patches_per_image = Config.PATCHES_PER_IMAGE if mode == "train" else 1

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Setup cache directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache", mode)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load data into memory
        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads images, processes them, and handles caching.
        Returns a list of dictionaries containing image data.
        """
        data_list = []

        for idx, row in self.df.iterrows():
            img_id = str(row["id"])

            # Define cache paths
            cache_path_noisy = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
            cache_path_clean = os.path.join(self.cache_dir, f"{img_id}_clean.npy")

            # Determine if we have labels (clean images)
            has_label = "label_path" in row and pd.notna(row["label_path"])

            # Try loading from cache
            loaded_from_cache = False
            if load_cached_data:
                if os.path.exists(cache_path_noisy):
                    if has_label:
                        if os.path.exists(cache_path_clean):
                            noisy_img = np.load(cache_path_noisy)
                            clean_img = np.load(cache_path_clean)
                            loaded_from_cache = True
                    else:
                        noisy_img = np.load(cache_path_noisy)
                        clean_img = None
                        loaded_from_cache = True

            # If not loaded from cache, process from scratch
            if not loaded_from_cache:
                # Load Noisy Image
                noisy_path = os.path.join(Config.INPUT_DIR, row["feature_path"])
                noisy_img = self._read_and_process_image(noisy_path)
                np.save(cache_path_noisy, noisy_img)

                # Load Clean Image (if available)
                if has_label:
                    clean_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    clean_img = self._read_and_process_image(clean_path)
                    np.save(cache_path_clean, clean_img)
                else:
                    clean_img = None

            data_list.append({"id": img_id, "noisy": noisy_img, "clean": clean_img})

        return data_list

    def _read_and_process_image(self, path):
        """
        Reads an image, converts to grayscale, and normalizes to [0, 1].
        """
        # Read as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Ensure channel dimension exists (H, W, 1)
        img = np.expand_dims(img, axis=-1)
        return img

    def __len__(self):
        """
        Returns the length of the dataset.
        For training, length is num_images * patches_per_image (Virtual Epoch).
        For val/test, length is num_images.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        """
        # Map global index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data[img_idx]

        noisy_img = sample["noisy"]
        clean_img = sample["clean"]
        img_id = sample["id"]

        if self.mode == "train":
            # --- High-Density Sampling & Augmentation ---

            h, w, _ = noisy_img.shape

            # Random Crop
            # Ensure image is larger than patch size (EDA confirms min dims > 128)
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                # Pad if necessary (though unlikely based on EDA)
                noisy_img = np.pad(
                    noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                )
                if clean_img is not None:
                    clean_img = np.pad(
                        clean_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                    )
                h, w, _ = noisy_img.shape

            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy_img[y : y + self.patch_size, x : x + self.patch_size, :]
            clean_patch = clean_img[y : y + self.patch_size, x : x + self.patch_size, :]

            # Augmentations
            # 1. Random Flips
            if np.random.rand() > 0.5:  # Horizontal
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)
            if np.random.rand() > 0.5:  # Vertical
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # 2. Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # Prepare for Tensor conversion
            # Copy to ensure negative strides from flips/rotations don't error in torch
            img_input = noisy_patch.copy()
            img_target = clean_patch.copy()

        else:
            # --- Validation / Test (Full Image) ---
            img_input = noisy_img
            if clean_img is not None:
                img_target = clean_img
            else:
                # Create dummy target for test set
                img_target = np.zeros_like(img_input)

        # Convert to Tensor (C, H, W)
        # Input is currently (H, W, C)
        tensor_input = torch.from_numpy(img_input.transpose((2, 0, 1))).float()
        tensor_target = torch.from_numpy(img_target.transpose((2, 0, 1))).float()

        return tensor_input, tensor_target, img_id
