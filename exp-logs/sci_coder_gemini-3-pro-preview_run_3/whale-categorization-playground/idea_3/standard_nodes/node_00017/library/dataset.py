import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import CFG


def get_transforms(data="train"):
    """
    Returns the albumentations transformations for the specified data split.

    Args:
        data (str): The mode of operation ('train', 'val', 'test').

    Returns:
        A.Compose: The composition of transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size),
                A.HorizontalFlip(p=0.5),
                # Strict geometric augmentations: Rotation, Shift, Scale
                # We avoid occlusion (Cutout/CoarseDropout) to preserve small scars/features
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif data == "val" or data == "test":
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class WhaleDataset(Dataset):
    def __init__(
        self,
        csv_file,
        mode="train",
        transform=None,
        id_map=None,
        exclude_new_whale=False,
    ):
        """
        Custom Dataset for Whale Species Identification.

        Args:
            csv_file (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transformations to apply.
            id_map (dict, optional): Mapping from class ID string to integer.
                                     If None and mode is train/val, it's created from the data.
            exclude_new_whale (bool): If True, rows with Id='new_whale' are dropped.
                                      Useful for training the classifier on known identities only.
        """
        self.csv_file = csv_file
        self.mode = mode
        self.transform = transform
        self.exclude_new_whale = exclude_new_whale

        # Load Metadata
        # Expecting columns: Image, Id (for train/val), file_path
        self.df = pd.read_csv(csv_file)

        # Filter out new_whale if requested (typically for training set)
        if self.exclude_new_whale and "Id" in self.df.columns:
            initial_len = len(self.df)
            self.df = self.df[self.df["Id"] != "new_whale"].reset_index(drop=True)
            # We silently filter; logging is handled by the caller if needed

        # Handle ID Mapping (Label Encoding)
        self.id_map = id_map
        if self.mode in ["train", "val"]:
            # If no map provided, create one from current data
            if self.id_map is None:
                # Sort unique IDs for deterministic mapping
                unique_ids = sorted(self.df["Id"].unique())
                self.id_map = {label: i for i, label in enumerate(unique_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train/img.jpg")
        # CFG.input_dir is "./input"
        image_path = os.path.join(CFG.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            # In a strict pipeline, this should raise an error
            raise FileNotFoundError(f"Image not found or corrupted: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode in ["train", "val"]:
            label_str = row["Id"]
            # Get integer label.
            # If label_str is not in id_map (e.g., 'new_whale' in validation set but not in training map),
            # we return -1. The validation loop must handle this (usually by ignoring for loss, or using for retrieval).
            label_idx = self.id_map.get(label_str, -1)

            return image, torch.tensor(label_idx, dtype=torch.long)

        elif self.mode == "test":
            # For test, we need the image name (Image ID) to create the submission file
            image_name = row["Image"]
            return image, image_name

    def get_id_map(self):
        """Returns the label encoding map used by this dataset."""
        return self.id_map
