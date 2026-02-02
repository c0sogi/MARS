import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class CadenceDataset(Dataset):
    """
    PyTorch Dataset for loading and processing Cadence Snippets.

    Loads all 6 cadence positions as channels for a single input tensor.
    Cite solution_lesson_node_00008: Stacking frames as channels allows explicit encoding of On-Off logic.
    Cite solution_lesson_node_00015: Centralize augmentation logic to prevent "Shadowed Pipelines".
    """

    def __init__(self, metadata_path, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_id = str(row["id"])
        target = float(row["target"])

        # Construct full file path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load data
        # Shape: (6, 273, 256) -> (Cadence Position, Frequency, Time)
        try:
            img = np.load(full_path).astype(np.float32)
        except FileNotFoundError:
            img = np.zeros((6, 273, 256), dtype=np.float32)

        # --- Preprocessing ---

        # 1. Min-Max Scale to [0, 1] per snippet
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 2. Transpose to (Height, Width, Channels) for Albumentations
        # Shape: (273, 256, 6)
        img = np.transpose(img, (1, 2, 0))

        # 3. Augmentations
        # We apply manual augmentations or use A.Compose.
        # Since we have 6 channels, we need to handle normalization carefully.
        # We will use manual resizing and flipping, and manual normalization.

        # Resize to Config.IMG_SIZE
        # Using Albumentations Resize is safe for N channels
        resize = A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1], p=1.0)
        img = resize(image=img)["image"]

        if self.mode == "train":
            # Horizontal Flip (Time Reversal)
            if np.random.rand() < 0.5:
                img = np.fliplr(img)

            # Vertical Flip (Frequency Inversion)
            # Cite solution_lesson_node_00014
            if np.random.rand() < 0.5:
                img = np.flipud(img)

        # Ensure contiguous array after flips
        img = np.ascontiguousarray(img)

        # 4. Normalize
        # ImageNet stats are for RGB. We repeat them for 6 channels.
        # Mean: [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        mean = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)

        img = (img - mean) / std

        # 5. ToTensor (Channels First)
        # (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        return {
            "input": torch.tensor(img, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "id": file_id,
        }
