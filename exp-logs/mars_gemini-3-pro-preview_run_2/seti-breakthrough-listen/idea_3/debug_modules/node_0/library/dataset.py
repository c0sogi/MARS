import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(data_type: str):
    """
    Returns the Albumentations transformation pipeline based on the data type.

    Args:
        data_type (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(
                    max_holes=Config.coarse_dropout_num_holes_max,
                    min_holes=Config.coarse_dropout_num_holes_min,
                    max_height=Config.coarse_dropout_hole_height,
                    max_width=Config.coarse_dropout_hole_width,
                    min_height=Config.coarse_dropout_hole_height // 2,
                    min_width=Config.coarse_dropout_hole_width // 2,
                    p=Config.coarse_dropout_prob,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class SETIDataset(Dataset):
    """
    PyTorch Dataset for SETI Signal Detection.
    Loads spectrograms, performs vertical stacking, instance normalization,
    and applies augmentations.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'target', and 'file_path'.
            transform (A.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.file_paths = df["file_path"].values
        self.targets = df["target"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # file_path in metadata is relative to input_root (e.g., "train/0/xxxx.npy")
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.input_root, rel_path)

        # Load data: Shape (6, 273, 256)
        try:
            image = np.load(full_path)
        except FileNotFoundError:
            # Fallback for robustness, though metadata validation ensures files exist
            # Create a zero array of expected shape
            image = np.zeros((6, 273, 256), dtype=np.float32)

        # 1. Vertical Stacking
        # Stack the 6 panels vertically to form (1638, 256)
        image = np.vstack(image).astype(np.float32)

        # 2. Instance Normalization
        # Normalize to zero mean and unit variance per sample
        mean = np.mean(image)
        std = np.std(image)
        # Avoid division by zero
        image = (image - mean) / (std + 1e-6)

        # 3. Prepare for Albumentations
        # Albumentations expects HWC or HW. We add a channel dim: (1638, 256, 1)
        image = image[..., np.newaxis]

        # 4. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (should be Tensor)
            converter = ToTensorV2()
            image = converter(image=image)["image"]

        # 5. Channel Expansion
        # Current shape is (1, 1638, 256) from ToTensorV2 (CHW)
        # Expand to 3 channels for pretrained backbone compatibility: (3, 1638, 256)
        image = image.repeat(Config.in_channels, 1, 1)

        # Get target
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, target
