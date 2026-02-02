import os
import cv2
import numpy as np
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import library.config as config
import library.utils as utils


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transformation pipeline for the specified split.

    Args:
        data_split (str): One of "train", "val", or "test".

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test only require resizing and conversion to tensor
        return A.Compose(
            [A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE), ToTensorV2()]
        )


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for loading BraTS MRI data.

    This dataset loads 3 specific slices (25%, 50%, 75% depth) from 4 modalities
    (FLAIR, T1w, T1wCE, T2w) to create a 12-channel 2.5D input volume.
    """

    def __init__(
        self, metadata, base_dir=config.INPUT_DIR, transform=None, is_test=False
    ):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing subject IDs and paths.
            base_dir (str): Root directory containing the input data.
            transform (A.Compose): Albumentations transforms.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.metadata = metadata
        self.base_dir = base_dir
        self.transform = transform
        self.is_test = is_test
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Initialize list to hold 12 slices (4 modalities * 3 slices)
        channels = []

        for mod in self.modalities:
            # Construct full path to the modality directory
            # Metadata contains relative paths like "train/00000/FLAIR"
            dir_path = os.path.join(self.base_dir, row[f"path_{mod}"])

            # Get all image files and sort them numerically
            # Files are named like "Image-1.dcm", "Image-10.dcm"
            try:
                files = os.listdir(dir_path)
                # Filter for .dcm files just in case
                files = [f for f in files if f.endswith(".dcm")]
                # Sort numerically by extracting the integer ID
                files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            except (FileNotFoundError, IndexError, ValueError):
                files = []

            num_files = len(files)

            # Determine indices for 25%, 50%, 75% depth
            if num_files > 0:
                indices = [
                    int(num_files * 0.25),
                    int(num_files * 0.50),
                    int(num_files * 0.75),
                ]
                # Clamp indices to be safe
                indices = [min(i, num_files - 1) for i in indices]
            else:
                indices = []

            # Load the selected slices
            for i in range(config.NUM_SLICES):
                if num_files > 0:
                    file_name = files[indices[i]]
                    file_path = os.path.join(dir_path, file_name)
                    # Use the provided raw loader
                    img = utils.load_dicom_raw(file_path)
                else:
                    # Fallback for missing directories/files
                    img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

                # Ensure image is resized to the target configuration size
                # load_dicom_raw might return 512x512 or 256x256
                if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
                    img = cv2.resize(
                        img,
                        (config.IMG_SIZE, config.IMG_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )

                channels.append(img)

        # Stack channels to create (H, W, C) array
        # C = 12 (3 slices * 4 modalities)
        image = np.stack(channels, axis=-1)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to ToTensorV2 logic if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return data based on split
        if self.is_test:
            # Return image and ID for submission mapping
            return image, row["BraTS21ID"]
        else:
            # Return image and label
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image, label
