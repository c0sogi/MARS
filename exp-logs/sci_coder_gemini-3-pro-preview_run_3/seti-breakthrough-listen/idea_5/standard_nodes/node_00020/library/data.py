import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import Config


def get_transforms(data_type="train"):
    """
    Returns the Albumentations composition of transforms.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    height, width = Config.IMAGE_SIZE

    # Base transforms: Resize and Normalize
    # We use ImageNet normalization stats as we are using a pretrained backbone.
    # Note: The dataset class performs MinMax scaling to [0, 1] before these transforms.
    transforms_list = [
        A.Resize(height=height, width=width),
    ]

    if data_type == "train":
        # Augmentations for training
        # HorizontalFlip (Time reversal) and VerticalFlip (Frequency inversion)
        # are valid for spectrograms and help robustness.
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )

    transforms_list.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=1.0,  # Input is scaled to [0, 1]
            ),
            ToTensorV2(),
        ]
    )

    # Define the composition with an additional target for the 'Off' stream
    # to ensure synchronized geometric transformations.
    return A.Compose(transforms_list, additional_targets={"image_off": "image"})


class TechnosignatureDataset(Dataset):
    """
    PyTorch Dataset for SETI Technosignature detection.

    Loads .npy cadence snippets, splits them into 'On' and 'Off' streams,
    performs MinMax scaling, and applies synchronized augmentations.
    """

    def __init__(self, metadata_path, data_type="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            data_type (str): 'train', 'val', or 'test'.
        """
        self.data_type = data_type
        self.metadata_path = metadata_path

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Initialize transforms
        self.transforms = get_transforms(data_type)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative path e.g., "train/0/xxxx.npy"
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load spectrogram: Shape (6, 273, 256)
        # Dimensions: (Cadence_Pos, Frequency, Time)
        try:
            data = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for missing files (should not happen based on metadata check)
            # Return a zero tensor to prevent crashing
            print(f"Error loading {file_path}: {e}")
            data = np.zeros((6, 273, 256), dtype=np.float32)

        # Split into On-Target and Off-Target streams
        # On-Target: A observations (indices 0, 2, 4)
        # Off-Target: B, C, D observations (indices 1, 3, 5)
        on_data = data[[0, 2, 4], :, :]  # Shape (3, 273, 256)
        off_data = data[[1, 3, 5], :, :]  # Shape (3, 273, 256)

        # Transpose to (H, W, C) for Albumentations -> (273, 256, 3)
        on_img = np.transpose(on_data, (1, 2, 0))
        off_img = np.transpose(off_data, (1, 2, 0))

        # Instance-wise Min-Max Scaling to [0, 1]
        # This is critical because raw spectrogram values can vary significantly.
        # We scale both streams based on the global min/max of the specific snippet
        # to preserve relative intensity differences between On and Off frames.
        global_min = min(on_img.min(), off_img.min())
        global_max = max(on_img.max(), off_img.max())
        denominator = global_max - global_min + 1e-6

        on_img = (on_img - global_min) / denominator
        off_img = (off_img - global_min) / denominator

        # Apply synchronized transforms
        # 'image' argument corresponds to on_img
        # 'image_off' argument corresponds to off_img (via additional_targets)
        transformed = self.transforms(image=on_img, image_off=off_img)

        image_on = transformed["image"]
        image_off = transformed["image_off"]

        # Get target
        if self.data_type == "test":
            target = torch.tensor(0.5, dtype=torch.float32)  # Placeholder
        else:
            target = torch.tensor(row["target"], dtype=torch.float32)

        return (image_on, image_off), target
