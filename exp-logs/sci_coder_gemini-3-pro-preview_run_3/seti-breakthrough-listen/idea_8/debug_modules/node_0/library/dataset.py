import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Signal Detection.
    Loads .npy spectrograms, applies padding to match model stride,
    performs synchronized augmentations across all 6 cadence channels,
    and normalizes using ImageNet statistics.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (A.Compose, optional): Custom augmentation pipeline.
        """
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Load appropriate metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_METADATA)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_METADATA)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Handle debugging subset
        if not Config.USE_FULL_DATA:
            # Use a small subset (e.g., 1000 samples) for quick debugging
            subset_size = min(len(self.df), 1000)
            self.df = self.df.iloc[:subset_size].reset_index(drop=True)

        # Initialize transforms
        if transform is None:
            self.transform = self.get_default_transforms(mode)
        else:
            self.transform = transform

    def get_default_transforms(self, mode):
        """
        Returns the Albumentations transformation pipeline.
        """
        # Standard ImageNet statistics
        # We repeat them for 6 channels because we treat the input as a 6-channel image
        # to ensure spatial augmentations are identical across all cadence positions.
        mean = [0.485, 0.456, 0.406] * 2  # [R, G, B, R, G, B]
        std = [0.229, 0.224, 0.225] * 2

        transforms_list = []

        if mode == "train":
            # Spatial augmentations for training
            # Applied to (H, W, C) where C=6
            transforms_list.extend(
                [
                    A.HorizontalFlip(p=0.5),  # Time reversal
                    A.VerticalFlip(p=0.5),  # Frequency inversion
                ]
            )

        # Normalization and Tensor conversion for all modes
        transforms_list.extend(
            [
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),  # Transposes (H, W, C) -> (C, H, W)
            ]
        )

        return A.Compose(transforms_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.df.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.input_dir, row["file_path"])

        try:
            # Load spectrogram: Shape (6, 273, 256) -> (Channels, Freq, Time)
            # Data is float16, convert to float32 for training
            image = np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Spectrogram file not found at: {file_path}")

        # --- Padding ---
        # Native shape: (6, 273, 256)
        # Target shape: (6, 288, 256)
        # We pad the Frequency dimension (axis 1) to be a multiple of 32 (288).
        # Pad width format: ((before_0, after_0), (before_1, after_1), (before_2, after_2))
        pad_freq = 288 - image.shape[1]  # Should be 15
        if pad_freq > 0:
            image = np.pad(
                image,
                ((0, 0), (0, pad_freq), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # --- Transpose for Albumentations ---
        # Albumentations expects (H, W, C).
        # Current: (C, H, W) -> (6, 288, 256)
        # Target: (288, 256, 6)
        image = np.transpose(image, (1, 2, 0))

        # --- Augmentation & Normalization ---
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns torch.Tensor (C, H, W)

        # --- Target ---
        if self.mode == "test":
            # Placeholder for test set
            target = torch.tensor(0.5, dtype=torch.float32)
        else:
            target = torch.tensor(row["target"], dtype=torch.float32)

        return image, target
