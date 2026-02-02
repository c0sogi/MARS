import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from library import config, utils


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'valid'.

    Returns:
        A.Compose: Composed transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Using Rotate as RandomRotate90 is fixed 90 degree steps.
                # The requirement specifies limiting to +/- 15 degrees.
                A.Rotate(limit=15, p=0.5),
            ]
        )
    else:
        # Validation/Test:
        # utils.read_dicom_robust already handles resizing to config.IMAGE_SIZE.
        # No further geometric transforms required.
        return A.Compose([])


class MGMTDataset(Dataset):
    def __init__(self, df, transforms=None, load_cached_anchors=True):
        """
        Dataset for MGMT promoter methylation prediction using MRI scans.

        Args:
            df (pd.DataFrame): Dataframe containing subject IDs, paths, and optionally labels.
            transforms (albumentations.Compose, optional): Transforms to apply to the image volume.
            load_cached_anchors (bool): Whether to load ROI anchors from the cache file.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

        # Load or compute ROI anchors using the utility function.
        # This function implements the required caching mechanism (loading from/saving to Parquet).
        self.anchors = utils.get_roi_anchors(
            self.df, load_cached_data=load_cached_anchors
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Retrieve the anchor slice index for this subject.
        # Default to 0 if not found (though get_roi_anchors ensures coverage).
        anchor_idx = self.anchors.get(subject_id, 0)

        # Load the 12-channel volume tensor: (12, H, W)
        # Structure: [FLAIR(z-5, z, z+5), T1w(...), T1wCE(...), T2w(...)]
        # utils.load_strided_volume handles reading, resizing, and stacking.
        volume_tensor = utils.load_strided_volume(row, anchor_idx)

        # Convert to Numpy (H, W, C) for Albumentations compatibility
        # volume_tensor is FloatTensor [0, 1]
        volume_np = volume_tensor.numpy().transpose(1, 2, 0)

        # Apply transforms if provided
        if self.transforms:
            augmented = self.transforms(image=volume_np)
            volume_np = augmented["image"]

        # Convert back to Tensor (C, H, W)
        # Ensure float32 data type
        volume_out = torch.from_numpy(volume_np.transpose(2, 0, 1)).float()

        # Get Target Label
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            # Placeholder for test set
            target = torch.tensor(-1.0, dtype=torch.float32)

        return volume_out, target
