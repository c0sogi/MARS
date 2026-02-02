import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import cyclic_roll, get_transforms


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.

    Supports:
    - Loading and caching of spectrogram images.
    - Cyclic Time-Rolling (Random for training, Fixed for TTA).
    - Pseudo-RGB conversion.
    - Soft Targets for Distillation.
    """

    def __init__(
        self,
        df,
        phase="train",
        model_name="resnet18",
        soft_targets=None,
        load_cached_data=True,
        fixed_shift_ratio=None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'rec_id', 'file_path_spec', and labels.
            phase (str): 'train', 'val', or 'test'. Controls augmentations.
            model_name (str): Name of the model (e.g., 'resnet18', 'densenet121') to determine input resolution.
            soft_targets (dict, optional): Dictionary mapping rec_id (int) to soft label numpy arrays (float).
                                           Used for distillation training.
            load_cached_data (bool): If True, attempts to load images from a cached .npy file.
            fixed_shift_ratio (float, optional): If provided, applies a deterministic cyclic time-shift.
                                                 Used for Test-Time Augmentation (TTA).
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.model_name = model_name
        self.soft_targets = soft_targets
        self.fixed_shift_ratio = fixed_shift_ratio

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

        # Initialize transforms
        self.transform = get_transforms(phase, model_name)

        # Load images into memory (with caching mechanism)
        self.images = self._load_images(load_cached_data)

    def _load_images(self, load_cached_data):
        """
        Loads images from disk or cache.

        Logic:
        1. Check if valid cache file exists in Config.CACHE_DIR.
        2. If yes and load_cached_data is True, load and return.
        3. Else, iterate through dataframe, load images using cv2.
        4. Handle missing images by creating zero-filled placeholders (robustness).
        5. Save processed array to cache.
        """
        # Create a unique cache filename based on phase and dataset size
        # This prevents collisions when using subsets for debugging
        cache_filename = f"images_{self.phase}_{len(self.df)}.npy"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                images = np.load(cache_path)
                # Verify length matches
                if len(images) == len(self.df):
                    return images
            except Exception:
                # If load fails, proceed to re-create
                pass

        # 2. Load from Scratch
        images = []
        # Standard dimensions based on EDA (Height=256, Width=1246)
        default_h, default_w = 256, 1246

        for _, row in self.df.iterrows():
            # Construct full path
            # Metadata contains relative path like "supplemental_data/spectrograms/..."
            # We enforce usage of Config.SPECTROGRAM_DIR (filtered spectrograms)
            filename = os.path.basename(row["file_path_spec"])
            full_path = os.path.join(Config.SPECTROGRAM_DIR, filename)

            img = None
            if os.path.exists(full_path):
                # Load image (cv2 reads as BGR or Grayscale depending on file)
                # Spectrograms are typically grayscale BMPs
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            # Handle missing or corrupt files
            if img is None:
                img = np.zeros((default_h, default_w), dtype=np.uint8)

            # Ensure 2D format (H, W) for storage efficiency
            if len(img.shape) == 3:
                img = img[:, :, 0]

            # Resize if dimensions differ significantly (though EDA suggests constant size)
            if img.shape[:2] != (default_h, default_w):
                img = cv2.resize(img, (default_w, default_h))

            images.append(img)

        images = np.array(images, dtype=np.uint8)

        # 3. Save Cache
        try:
            np.save(cache_path, images)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image from memory
        # Shape: (H, W)
        image = self.images[idx].copy()

        # --- 1. Cyclic Rolling (Time-Shift) ---
        # Strategy:
        # - Train: Random Roll [0, 1]
        # - TTA: Fixed Roll (e.g., 0.25, 0.5)
        # - Val/Test (Standard): No Roll

        shift = 0.0
        if self.fixed_shift_ratio is not None:
            shift = self.fixed_shift_ratio
        elif self.phase == "train":
            shift = np.random.rand()

        # Prepare for cyclic_roll (expects H, W, C)
        image = image[:, :, np.newaxis]

        if shift > 0.0:
            image = cyclic_roll(image, shift_ratio=shift)

        # --- 2. Pseudo-RGB Conversion ---
        # Replicate the single channel 3 times to match ImageNet pretrained input
        # Shape: (H, W, 1) -> (H, W, 3)
        image = np.repeat(image, 3, axis=2)

        # --- 3. Albumentations Transforms ---
        # Applies Resize, Normalize, and SpecAugment (if train)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # --- 4. Labels & Metadata ---
        rec_id = self.df.iloc[idx]["rec_id"]

        # Hard Labels (Ground Truth)
        # For test set, these are placeholders (zeros)
        labels = self.df.iloc[idx][self.label_cols].values.astype(np.float32)

        # Soft Targets (Distillation)
        # If soft_targets dict is provided, look up the probability vector
        if self.soft_targets is not None:
            if rec_id in self.soft_targets:
                soft_labels = self.soft_targets[rec_id].astype(np.float32)
            else:
                # Fallback to hard labels if soft target missing for this ID
                soft_labels = labels
        else:
            # If no distillation, soft_labels are just the hard labels
            soft_labels = labels

        return {
            "image": image,  # Tensor (3, H, W)
            "labels": torch.tensor(labels),  # Tensor (19,)
            "soft_labels": torch.tensor(soft_labels),  # Tensor (19,)
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }
