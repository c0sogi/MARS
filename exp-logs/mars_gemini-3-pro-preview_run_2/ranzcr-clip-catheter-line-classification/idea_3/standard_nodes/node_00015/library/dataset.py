import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data="train"):
    """
    Returns the transformation pipeline.

    Strategy Compliance:
    - Resolution: 640x640
    - Sequence: Resize (Aspect Preserved) -> CLAHE -> Augmentations -> Pad
    """
    transforms = [
        # 1. Aspect-Ratio Preserving Resize
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
        # 2. Intensity Transformation (CLAHE)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ]

    if data == "train":
        # Destructive Augmentations (Cite solution_lesson_node_00014)
        transforms.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )

    transforms.extend(
        [
            # 3. Padding
            A.PadIfNeeded(
                min_height=Config.IMAGE_SIZE,
                min_width=Config.IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            # Standard ImageNet Normalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # Convert to PyTorch Tensor
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter and Line Position Detection.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (paths and labels).
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-compute full file paths
        # Metadata contains relative paths (e.g., "train/uid.jpg")
        # We join with INPUT_DIR to get absolute paths.
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in self.df["file_path"].values
        ]

        # Extract labels for training/validation
        if self.mode != "test":
            self.labels = self.df[Config.TARGET_COLS].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        # Robustness check: if image fails to load, create a blank image
        # This prevents the entire training run from crashing due to a single bad file
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # OpenCV loads as BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, we return the image and a dummy label
            # The dummy label ensures compatibility with standard training loops
            return image, torch.tensor(0)
        else:
            # For train/val, return image and multi-label targets
            label = torch.tensor(self.labels[idx])
            return image, label
