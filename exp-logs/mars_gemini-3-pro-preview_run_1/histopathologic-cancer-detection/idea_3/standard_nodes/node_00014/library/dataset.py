import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(split: str):
    """
    Generates the Albumentations transformation pipeline for a given dataset split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    transforms_list = []

    # 1. Hard Attention via ROI Cropping
    # Extract the central 48x48 region as defined in Config.
    # This removes background noise and focuses on the target area.
    transforms_list.append(
        A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE)
    )

    # 2. Augmentations (Train only)
    if split == "train":
        # Geometric Augmentations
        transforms_list.extend(
            [
                A.HorizontalFlip(p=Config.PROB_HFLIP),
                A.VerticalFlip(p=Config.PROB_VFLIP),
                A.RandomRotate90(p=Config.PROB_ROTATE),
            ]
        )

        # Color Augmentations (Restricted)
        # We apply mild brightness and contrast changes.
        # Hue and Saturation are excluded to preserve H&E stain semantics.
        if Config.COLOR_JITTER_BRIGHTNESS > 0 or Config.COLOR_JITTER_CONTRAST > 0:
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=Config.COLOR_JITTER_BRIGHTNESS,
                    contrast_limit=Config.COLOR_JITTER_CONTRAST,
                    brightness_by_max=False,
                    p=0.5,
                )
            )

    # 3. Normalization and Tensor Conversion
    # Normalize using ImageNet statistics and convert to PyTorch Tensor (CHW)
    transforms_list.extend(
        [A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD), ToTensorV2()]
    )

    return A.Compose(transforms_list)


class TumorDataset(Dataset):
    """
    PyTorch Dataset for loading digital pathology images.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and optionally 'label'.
            transforms (albumentations.Compose, optional): Transformations to apply.
        """
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # metadata contains relative paths like "train/id.tif"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Robustness check: Handle missing or corrupt files
        if image is None:
            # Return a black image of the original size (96x96)
            # This ensures the dataloader doesn't crash
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Handle label
        if "label" in row:
            # Return float32 for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # Return dummy label for test set
            return image, torch.tensor(0.0, dtype=torch.float32)
