import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from library.config import Config


def get_transforms(mode, img_size):
    """
    Returns the appropriate composition of transforms for the given mode and image size.
    Implements Context-Preserving Augmentation and Resolution Diversity.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): Target spatial dimension (e.g., 224, 256, 288).

    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: Random Crop with high scale range
                transforms.RandomResizedCrop(
                    (img_size, img_size),
                    scale=Config.AUG_RRC_SCALE,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Photometric distortion
                transforms.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=Config.AUG_COLOR_JITTER,
                    hue=0.0,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Deterministic resizing for Validation and Test
        return transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogCatDataset(Dataset):
    """
    Dataset class for Dog vs Cat classification.
    Supports loading images from disk, applying transforms, and handling both
    hard labels (int) and soft pseudo-labels (float) for Noisy Student training.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'filepath' and 'label' (or 'id').
            transform (callable, optional): Transform to apply to the image.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        rel_path = row["filepath"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # Use cv2 for robust loading, then convert to PIL for torchvision compatibility
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for potentially corrupt/missing images
            # Create a blank image to ensure the pipeline doesn't crash
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.mode == "test":
            # For test set, we need the ID for submission mapping
            img_id = row["id"]
            return image, img_id
        else:
            # For train/val, we return the label
            # Supports both hard (0/1) and soft (0.0-1.0) labels
            label = row["label"]
            # Convert to float32 tensor for BCEWithLogitsLoss
            target = torch.tensor(label, dtype=torch.float32)
            return image, target
