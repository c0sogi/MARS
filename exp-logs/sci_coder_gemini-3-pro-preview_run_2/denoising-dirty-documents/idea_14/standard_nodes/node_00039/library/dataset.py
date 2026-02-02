import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Implements high-density sampling (random patches) for training and
    full-image loading for validation/testing.
    """

    def __init__(
        self,
        metadata_path,
        root_dir=Config.INPUT_DIR,
        augment=False,
        train_mode=True,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing input images.
            augment (bool): Whether to apply geometric augmentations.
            train_mode (bool): If True, uses high-density patch sampling.
                               If False, returns full images for inference.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.augment = augment
        self.train_mode = train_mode
        self.patch_size = Config.PATCH_SIZE
        # Use configured patches per image for training, 1 for validation/test
        self.patches_per_image = Config.PATCHES_PER_IMAGE if train_mode else 1

        self.cache_dir = Config.CACHE_DIR
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        self.data = []
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads images into memory, utilizing caching to speed up subsequent runs.
        """
        for _, row in self.metadata.iterrows():
            img_id = str(row["id"])

            # Define cache paths
            cache_noisy_path = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
            cache_clean_path = os.path.join(self.cache_dir, f"{img_id}_clean.npy")

            # Determine if we expect a label (clean image)
            has_label = "label_path" in row and pd.notna(row["label_path"])

            loaded = False
            noisy_img = None
            clean_img = None

            # 1. Try loading from cache
            if load_cached_data:
                if os.path.exists(cache_noisy_path):
                    if has_label:
                        if os.path.exists(cache_clean_path):
                            noisy_img = np.load(cache_noisy_path)
                            clean_img = np.load(cache_clean_path)
                            loaded = True
                    else:
                        noisy_img = np.load(cache_noisy_path)
                        loaded = True

            # 2. If not loaded, process from source
            if not loaded:
                # Load Noisy Image
                noisy_full_path = os.path.join(self.root_dir, row["feature_path"])
                noisy_raw = cv2.imread(noisy_full_path, cv2.IMREAD_GRAYSCALE)
                if noisy_raw is None:
                    raise FileNotFoundError(f"Image not found: {noisy_full_path}")
                # Normalize to [0, 1]
                noisy_img = noisy_raw.astype(np.float32) / 255.0
                # Save to cache
                np.save(cache_noisy_path, noisy_img)

                # Load Clean Image (if available)
                if has_label:
                    clean_full_path = os.path.join(self.root_dir, row["label_path"])
                    clean_raw = cv2.imread(clean_full_path, cv2.IMREAD_GRAYSCALE)
                    if clean_raw is None:
                        raise FileNotFoundError(f"Image not found: {clean_full_path}")
                    # Normalize to [0, 1]
                    clean_img = clean_raw.astype(np.float32) / 255.0
                    # Save to cache
                    np.save(cache_clean_path, clean_img)

            self.data.append({"id": img_id, "noisy": noisy_img, "clean": clean_img})

    def __len__(self):
        """
        Returns the total number of samples.
        In train mode, this is num_images * patches_per_image.
        In val/test mode, this is num_images.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        Train mode: Returns a random augmented patch.
        Val/Test mode: Returns the full image.
        """
        # Map linear index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data[img_idx]

        noisy = sample["noisy"]
        clean = sample["clean"]

        if self.train_mode:
            # --- Training: Random Patch Extraction & Augmentation ---
            h, w = noisy.shape

            # Handle cases where image is smaller than patch size (unlikely per EDA, but safe)
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                if clean is not None:
                    clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Random crop coordinates
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy[
                top : top + self.patch_size, left : left + self.patch_size
            ]
            clean_patch = (
                clean[top : top + self.patch_size, left : left + self.patch_size]
                if clean is not None
                else None
            )

            # Geometric Augmentations
            if self.augment:
                # Random Horizontal/Vertical Flip
                if np.random.rand() > 0.5:
                    noisy_patch = np.flip(noisy_patch, axis=0)
                    if clean_patch is not None:
                        clean_patch = np.flip(clean_patch, axis=0)
                if np.random.rand() > 0.5:
                    noisy_patch = np.flip(noisy_patch, axis=1)
                    if clean_patch is not None:
                        clean_patch = np.flip(clean_patch, axis=1)

                # Random Rotation (0, 90, 180, 270 degrees)
                k = np.random.randint(0, 4)
                if k > 0:
                    noisy_patch = np.rot90(noisy_patch, k)
                    if clean_patch is not None:
                        clean_patch = np.rot90(clean_patch, k)

            # Convert to Tensor (C, H, W)
            # Use .copy() to ensure memory is contiguous (needed after flip/rot)
            noisy_t = torch.from_numpy(noisy_patch.copy()).unsqueeze(0)
            clean_t = (
                torch.from_numpy(clean_patch.copy()).unsqueeze(0)
                if clean_patch is not None
                else None
            )

            return noisy_t, clean_t

        else:
            # --- Validation/Test: Full Image ---
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)
            clean_t = (
                torch.from_numpy(clean).unsqueeze(0)
                if clean is not None
                else torch.tensor([])
            )

            return noisy_t, clean_t, sample["id"]
