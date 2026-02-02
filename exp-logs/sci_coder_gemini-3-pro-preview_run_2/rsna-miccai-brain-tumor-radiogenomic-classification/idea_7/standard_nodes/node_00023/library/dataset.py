import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import load_metadata
from library.image_utils import get_roi_cache, load_patient_volume


def get_transforms(split: str):
    """
    Returns the appropriate Albumentations transform pipeline for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Limit rotation to +/- 10 degrees as per requirements
                A.Rotate(limit=10, p=0.5, border_mode=0, value=0),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we just convert to tensor
        return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma MGMT promoter methylation prediction.

    Loads 12-channel 2.5D volumes (3 slices per 4 modalities) using robust
    Z-profile anchoring.
    """

    def __init__(self, split="train", transform=None, load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None, defaults are generated based on split.
            load_cached_data (bool): Whether to load anchor indices from cache.
        """
        self.split = split
        self.df = load_metadata(split)

        # Initialize ROI cache for anchor indices
        # This ensures consistent slice selection across epochs
        self.roi_cache = get_roi_cache(self.df, load_cached_data=load_cached_data)

        # Set transforms
        if transform is None:
            self.transform = get_transforms(split)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # Retrieve anchor index from cache (defaults to 0 if missing, though cache handles this)
        anchor_index = self.roi_cache.get(brats_id, 0)

        # Load volume: Returns torch.Tensor of shape (12, H, W) with values 0-1
        volume_tensor = load_patient_volume(row, anchor_index)

        # Convert to Numpy (H, W, C) for Albumentations
        # load_patient_volume returns (C, H, W), so we transpose
        volume_np = volume_tensor.numpy()
        volume_np = np.transpose(volume_np, (1, 2, 0))  # (H, W, 12)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=volume_np)
            image = augmented[
                "image"
            ]  # Returns Tensor (12, H, W) if ToTensorV2 is used
        else:
            # Fallback if no transform provided (should not happen with default init)
            image = torch.from_numpy(np.transpose(volume_np, (2, 0, 1)))

        # Return logic based on split
        if self.split == "test":
            # For test, we need the ID to map predictions for submission
            return image, brats_id
        else:
            # For train/val, we return the label
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32).unsqueeze(0)
            return image, label
