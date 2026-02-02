import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(data_type: str):
    """
    Returns the Albumentations composition of transforms based on the data type.

    Pipeline:
    1. Resize Longest Edge -> 640
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. (Train Only) Geometric & Destructive Augmentations
    4. Pad to Square (640x640)
    5. Normalize & ToTensor

    Args:
        data_type (str): 'train' or 'valid' (applies to validation and test).
    """
    img_size = Config.image_size

    # Base transforms applied to all sets (Preprocessing)
    # 1. Resize preserving aspect ratio
    transforms_list = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_CUBIC),
    ]

    # 2. Intensity Transformation (CLAHE) - Applied to both train and test
    # as per the "Inference" section of the Idea.
    transforms_list.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0))

    if data_type == "train":
        # 3. Regularization / Augmentation
        transforms_list.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=Config.aug_prob,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    min_holes=1,
                    min_height=int(img_size * 0.05),
                    min_width=int(img_size * 0.05),
                    fill_value=0,
                    p=Config.aug_prob,
                ),
            ]
        )

    # 4. Pad to Square
    transforms_list.append(
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        )
    )

    # 5. Normalize & Tensor
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter/Line Detection.
    """

    def __init__(self, df: pd.DataFrame, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, targets).
            transform (albumentations.Compose): Transforms to apply.
            is_test (bool): If True, returns dummy labels.
        """
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.target_cols = Config.target_cols

        # Pre-extract file paths and labels to avoid dataframe overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.labels = self.df[self.target_cols].values.astype(np.float32)
        else:
            # Dummy labels for test set
            self.labels = np.zeros(
                (len(self.df), len(self.target_cols)), dtype=np.float32
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # Config.image_base_dir is "./input", file_path is "train/xxx.jpg"
        img_path = os.path.join(Config.image_base_dir, self.file_paths[idx])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though verification script showed 0 missing)
            # Create a black image to prevent crashing
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in this pipeline)
            transforms = get_transforms("valid")
            augmented = transforms(image=image)
            image = augmented["image"]

        # Get labels
        label = torch.tensor(self.labels[idx])

        return image, label
