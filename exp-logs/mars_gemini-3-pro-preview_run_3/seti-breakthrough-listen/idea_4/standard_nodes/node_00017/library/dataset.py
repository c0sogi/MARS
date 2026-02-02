import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train", img_size=(224, 224)):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (tuple): Target (height, width).

    Returns:
        A.Compose: The composition of transforms.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms_list = [
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    ]

    if mode == "train":
        # Horizontal Flip (Time Reversal)
        transforms_list.append(A.HorizontalFlip(p=0.5))
        # Vertical Flip (Frequency Inversion) - Cite Lesson 14
        transforms_list.append(A.VerticalFlip(p=0.5))

    transforms_list.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )

    # Use additional_targets to ensure 'image_off' receives the exact same
    # geometric transformations (Resize, Flips) as 'image'. - Cite Lesson 15
    return A.Compose(transforms_list, additional_targets={"image_off": "image"})


class CadenceDataset(Dataset):
    """
    PyTorch Dataset for loading and processing Cadence Snippets.

    Splits the 6-channel input into two 3-channel streams:
    - On-Target (A observations): Indices 0, 2, 4
    - Off-Target (B, C, D observations): Indices 1, 3, 5
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None, default transforms are generated.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)

        # Use default transforms if none provided
        if transform is None:
            self.transform = get_transforms(mode=mode, img_size=Config.IMG_SIZE)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_id = str(row["id"])
        target = float(row["target"])

        # Construct full file path
        # metadata 'file_path' is relative to input dir (e.g., "train/0/000....npy")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load data
        # Shape: (6, 273, 256) -> (Cadence Position, Frequency, Time)
        try:
            img = np.load(full_path).astype(np.float32)
        except FileNotFoundError:
            # Fallback for robustness (should not happen given verification)
            # Create a zero array of expected shape
            img = np.zeros((6, 273, 256), dtype=np.float32)

        # --- Preprocessing ---

        # 1. Min-Max Scale to [0, 1] per snippet
        # This preserves the relative intensity of the signal vs noise within the snippet
        # while mapping it to the range expected by ImageNet normalization.
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 2. Split into On-Target and Off-Target streams
        # On-Target: A, A, A (indices 0, 2, 4)
        # Off-Target: B, C, D (indices 1, 3, 5)
        on_target = img[[0, 2, 4], :, :]  # Shape: (3, 273, 256)
        off_target = img[[1, 3, 5], :, :]  # Shape: (3, 273, 256)

        # 3. Transpose to (Height, Width, Channels) for Albumentations
        # Resulting Shape: (273, 256, 3)
        on_target = np.transpose(on_target, (1, 2, 0))
        off_target = np.transpose(off_target, (1, 2, 0))

        # 4. Apply Transforms
        # Use the configured Albumentations pipeline which handles synchronization
        # via 'additional_targets'. - Cite Lesson 15
        augmented = self.transform(image=on_target, image_off=off_target)

        return {
            "on_input": augmented["image"],
            "off_input": augmented["image_off"],
            "target": torch.tensor(target, dtype=torch.float32),
            "id": file_id,
        }
