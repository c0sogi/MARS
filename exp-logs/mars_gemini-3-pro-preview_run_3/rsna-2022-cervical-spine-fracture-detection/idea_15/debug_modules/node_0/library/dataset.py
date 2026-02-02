import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.dicom_utils import process_scan, get_25d_stack


def get_transforms(split="train"):
    """
    Returns the albumentations transform pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose or A.ReplayCompose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        # Use ReplayCompose to ensure the same random geometric transform
        # is applied to every slice in the volume.
        return A.ReplayCompose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # For validation/test, only normalize.
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


class CervicalSpineDataset(Dataset):
    """
    PyTorch Dataset for Cervical Spine Fracture Detection.
    Loads cached 3D volumes, samples slices, applies consistent 2.5D stacking
    and deterministic volumetric augmentation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        images_dir: str,
        transform=None,
        split: str = "train",
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing StudyInstanceUID and labels.
            images_dir (str): Path to the directory containing study folders.
            transform (albumentations.Compose, optional): Transform pipeline.
            split (str): 'train', 'val', or 'test'. Used to determine behavior.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.split = split
        self.num_slices = Config.NUM_SLICES

        # Define target columns
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # 1. Load Volume
        # process_scan handles caching logic (load .npy if exists, else process raw)
        # It returns a uint8 volume of shape (Depth, H, W)
        try:
            volume = process_scan(study_uid, self.images_dir, load_cached_data=True)
        except Exception as e:
            # Fallback for corrupt/missing data: generate a black volume
            # This prevents the training loop from crashing
            volume = np.zeros(
                (self.num_slices, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
            )

        depth = volume.shape[0]

        # 2. Uniform Sampling
        # We need exactly self.num_slices.
        # If depth < num_slices, this repeats indices.
        # If depth > num_slices, this subsamples.
        if depth > 0:
            indices = np.linspace(0, depth - 1, num=self.num_slices).round().astype(int)
        else:
            indices = np.zeros(self.num_slices, dtype=int)

        # 3. Process Slices (2.5D Stacking + Augmentation)
        stacked_images = []

        # We need to handle deterministic replay for training
        replay_data = None

        for i, slice_idx in enumerate(indices):
            # Extract 3-channel stack: [z-1, z, z+1]
            # Shape: (H, W, 3)
            img_stack = get_25d_stack(volume, slice_idx)

            if self.transform:
                if isinstance(self.transform, A.ReplayCompose):
                    # For ReplayCompose (Training):
                    if i == 0:
                        # Apply transform to the first slice and save parameters
                        augmented = self.transform(image=img_stack)
                        replay_data = augmented["replay"]
                        img_tensor = augmented["image"]
                    else:
                        # Replay the exact same transform for subsequent slices
                        augmented = self.transform.replay(replay_data, image=img_stack)
                        img_tensor = augmented["image"]
                else:
                    # For standard Compose (Validation/Test)
                    augmented = self.transform(image=img_stack)
                    img_tensor = augmented["image"]
            else:
                # Fallback if no transform provided
                transforms = ToTensorV2()
                img_tensor = transforms(image=img_stack)["image"]

            stacked_images.append(img_tensor)

        # Stack into a single tensor: (Num_Slices, Channels, H, W)
        # Shape: (64, 3, 224, 224)
        input_tensor = torch.stack(stacked_images)

        # 4. Prepare Targets
        targets = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        # Check if target columns exist (Training/Validation)
        if all(col in self.df.columns for col in self.target_cols):
            label_values = row[self.target_cols].values.astype(np.float32)
            targets = torch.tensor(label_values, dtype=torch.float32)

        return {"image": input_tensor, "targets": targets, "study_uid": study_uid}
