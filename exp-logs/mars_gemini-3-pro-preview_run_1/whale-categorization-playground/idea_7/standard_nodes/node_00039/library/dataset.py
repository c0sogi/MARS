import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import WhaleLabelEncoder


def get_transforms(phase, image_size):
    """
    Returns the Albumentations transform pipeline for the given phase and image size.

    Args:
        phase (str): 'train', 'val', or 'test'.
        image_size (int): Target resolution (e.g., 256, 320).

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                # Conservative Geometric Augmentations
                # Rotation +/- 20 degrees, Scale 0.9-1.1, No translation (shift=0)
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # Photometric Augmentations
                # Only Brightness and Contrast, explicitly avoiding Hue/Saturation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    def __init__(
        self,
        csv_path,
        label_encoder=None,
        transform=None,
        debug=False,
        load_cached_data=True,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            label_encoder (WhaleLabelEncoder, optional): Fitted encoder.
                If None and 'Id' is in CSV, a new encoder is created and fitted on this data.
                NOTE: For validation sets, always pass the encoder fitted on the training set.
            transform (A.Compose, optional): Albumentations transform pipeline.
            debug (bool): If True, subsamples the dataset for quick debugging.
            load_cached_data (bool): Whether to attempt loading cached encoder data.
        """
        self.csv_path = csv_path
        self.transform = transform
        self.debug = debug

        # Load Metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Handle Debugging
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=42
            ).reset_index(drop=True)

        # Determine if we have labels (Train/Val) or not (Test)
        self.has_labels = "Id" in self.df.columns
        self.label_encoder = label_encoder

        if self.has_labels:
            # If no encoder provided, create and fit one (typically done for the training set)
            if self.label_encoder is None:
                self.label_encoder = WhaleLabelEncoder()
                # Fit on the current dataframe's IDs
                # The cache path is handled internally by WhaleLabelEncoder in library.utils
                self.label_encoder.fit(
                    self.df["Id"].values, load_cached_data=load_cached_data
                )
            else:
                # Filter out samples with unknown labels (Cite debug_lesson_1)
                # This is crucial for debug mode where random subsampling breaks the
                # "validation is subset of train" invariant.
                known_classes = set(self.label_encoder.classes_)
                self.df = self.df[self.df["Id"].isin(known_classes)].reset_index(
                    drop=True
                )

            # Pre-encode labels to integer indices
            # This handles 'new_whale' and specific IDs uniformly
            self.labels = self.label_encoder.transform(self.df["Id"].values)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to INPUT_DIR (e.g., 'train/00022e1a.jpg')
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Robustness check: return a black image if file read fails
            # This prevents the entire training loop from crashing due to one bad file
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            t = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
            image = t(image=image)["image"]

        # Prepare Return Values
        image_filename = row["Image"]

        if self.has_labels:
            target = self.labels[idx]
            # Return image, integer label, and filename
            return image, torch.tensor(target, dtype=torch.long), image_filename
        else:
            # For test set, return -1 as target
            return image, torch.tensor(-1, dtype=torch.long), image_filename
