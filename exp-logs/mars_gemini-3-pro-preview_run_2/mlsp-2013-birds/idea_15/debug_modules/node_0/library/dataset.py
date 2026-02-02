import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class BirdDataset(Dataset):
    """
    PyTorch Dataset for the Bird Species Classification task.

    Features:
    - Loads filtered spectrograms (BMP).
    - Supports in-memory preloading for small datasets to speed up training.
    - Applies Albumentations transforms (including custom TimeRolling).
    - Returns multi-hot encoded labels.
    """

    def __init__(self, csv_path, transform=None, debug=False, preload=True):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train, val, or test).
            transform (A.Compose, optional): Albumentations transform pipeline.
            debug (bool): If True, restricts dataset size for debugging.
            preload (bool): If True, loads all images into RAM.
        """
        self.csv_path = csv_path
        self.transform = transform
        self.preload = preload
        self.image_dir = Config.IMAGE_DIR

        # Load Metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Apply Debugging Constraints
        if debug or Config.DEBUG:
            limit = Config.DEBUG_SAMPLES
            self.df = self.df.head(limit).reset_index(drop=True)

        # Identify Label Columns (species_0 to species_18)
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]

        # Pre-convert labels and IDs to numpy for fast access
        # Ensure labels are float32 for BCEWithLogitsLoss
        self.labels = self.df[self.label_cols].values.astype(np.float32)
        self.rec_ids = self.df["rec_id"].values.astype(np.int64)

        # Extract filenames
        # Metadata 'file_path_spec' points to 'spectrograms/PC...bmp'
        # We need to map this to 'filtered_spectrograms/PC...bmp'
        self.filenames = self.df["file_path_spec"].apply(os.path.basename).tolist()

        # Preload Images into Memory
        self.images = None
        if self.preload:
            self.images = []
            for fname in self.filenames:
                path = os.path.join(self.image_dir, fname)
                self.images.append(self._load_image(path))

    def _load_image(self, path):
        """
        Helper to load an image using OpenCV.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file missing: {path}")

        # Load image unchanged (likely grayscale BMP)
        # Channel conversion to RGB is handled in transforms.py
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image (corrupt or invalid): {path}")

        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Transformed image tensor.
            label (torch.Tensor): Multi-hot label tensor.
            rec_id (torch.Tensor): Recording ID.
        """
        # 1. Get Image
        if self.preload:
            image = self.images[idx]
        else:
            fname = self.filenames[idx]
            path = os.path.join(self.image_dir, fname)
            image = self._load_image(path)

        # 2. Apply Transforms
        if self.transform:
            # Albumentations expects 'image' keyword argument
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor if no transform provided
            image = torch.from_numpy(image).float()
            # Add channel dim if missing (H, W) -> (C, H, W)
            if image.ndim == 2:
                image = image.unsqueeze(0)
            elif image.ndim == 3:
                image = image.permute(2, 0, 1)

        # 3. Get Label and ID
        label = self.labels[idx]
        rec_id = self.rec_ids[idx]

        return image, torch.tensor(label), torch.tensor(rec_id)
