import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import from the provided library files
from library.config import IMG_SIZE
from library.data_processing import get_processed_dataframe, load_subject_volume


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Strategy: Spatially-Preserved Augmentation.
    - We strictly EXCLUDE RandomScale and Shift (Translation) to preserve the
      spatial priors established by the Center-of-Mass alignment.
    - We include geometric distortions (Elastic, Grid) and orientation changes (Flip, Rotate)
      to improve robustness without breaking the volumetric alignment.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate without scaling or shifting. border_mode=0 (constant) with value 0
                A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                # Spatial distortions
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Convert back to tensor. Note: ToTensorV2 handles HWC -> CHW
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we only convert to tensor.
        # Normalization is already handled in load_subject_volume.
        return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the Centroid-Aligned Scale-Invariant Volumetric (CASIV) strategy.

    Loads 9-channel volumetric slabs based on precomputed Center-of-Mass statistics.
    """

    def __init__(self, split="train", transform=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use cached stats/metadata.
        """
        self.split = split
        self.transform = transform

        # Load the dataframe which merges metadata (paths) and stats (CoM, depth)
        self.df = get_processed_dataframe(
            split=split, load_cached_data=load_cached_data
        )

        # Extract labels if available (train/val)
        if "MGMT_value" in self.df.columns:
            self.labels = self.df["MGMT_value"].values.astype(np.float32)
        else:
            self.labels = None

        # Extract IDs for submission
        self.ids = self.df["BraTS21ID"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row containing paths and stats
        row = self.df.iloc[idx]

        # Load the 9-channel volume tensor (C, H, W)
        # load_subject_volume returns a torch.Tensor
        # We pass 'row' for both subject_row and stats_row arguments as they are merged
        volume_tensor = load_subject_volume(row, row)

        # Convert to Numpy (H, W, C) for Albumentations
        # volume_tensor is (C, H, W), permute to (H, W, C)
        image_np = volume_tensor.permute(1, 2, 0).numpy()

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_np)
            # ToTensorV2 converts back to (C, H, W) and returns a tensor
            image_tensor = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen with get_transforms)
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

        # Return appropriate data based on split
        if self.split == "test":
            # For test, we need the ID to map predictions
            return image_tensor, self.ids[idx]
        else:
            # For train/val, we return image and label
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, label
