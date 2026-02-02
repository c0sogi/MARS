import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(data_type="train"):
    """
    Constructs the data augmentation and preprocessing pipeline.

    Sequence:
    1. Resize (Longest Edge) -> Preserves aspect ratio.
    2. CLAHE -> Enhances contrast on valid pixels before padding.
    3. Pad (Square) -> Zero padding to reach 640x640.
    4. Augment (Train only) -> ShiftScaleRotate, CoarseDropout.
    5. Normalize -> ImageNet stats.
    6. ToTensor -> Convert to PyTorch tensor.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    transforms = []

    # 1. Aspect-Ratio Preserving Resize
    # Resizes the longest edge to Config.image_size (640), scaling the other dimension proportionally.
    transforms.append(A.LongestMaxSize(max_size=Config.image_size, always_apply=True))

    # 2. Intensity Transformation (CLAHE)
    # Applied before padding to prevent boundary artifacts and ensure histogram is calculated on image content.
    if Config.use_clahe:
        transforms.append(
            A.CLAHE(
                clip_limit=Config.clahe_clip_limit,
                tile_grid_size=Config.clahe_tile_grid_size,
                p=1.0,  # Always apply as per strategy
            )
        )

    # 3. Padding to Square
    # Pads the shorter dimension with zeros to achieve a square image.
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.image_size,
            min_width=Config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            always_apply=True,
        )
    )

    # 4. Augmentations (Train only)
    if data_type == "train":
        # Geometric robustness
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            )
        )
        # Regularization via information dropping
        transforms.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=int(Config.image_size * 0.05),  # ~32px
                max_width=int(Config.image_size * 0.05),  # ~32px
                min_holes=1,
                fill_value=0,
                p=0.5,
            )
        )

    # 5. Normalization and Tensor conversion
    transforms.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=True,
        )
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter Detection.
    Reads images based on metadata paths and applies the strict preprocessing pipeline.
    """

    def __init__(self, metadata_path, transform=None, is_test=False, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Transforms to apply.
            is_test (bool): If True, returns (image, study_uid). If False, returns (image, labels).
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.input_dir
        self.target_cols = Config.target_cols

        # Load Data
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        if debug:
            self.df = self.df.iloc[:100].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train/1.2.3.jpg")
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Read Image
        image = cv2.imread(image_path)

        if image is None:
            # Fallback for safety, though metadata verification should prevent this
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal fallback transform
            fallback = A.Compose(
                [
                    A.Resize(Config.image_size, Config.image_size),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = fallback(image=image)["image"]

        # Return Data
        if self.is_test:
            # For inference, we need the ID to map predictions to the submission file
            return image, row["StudyInstanceUID"]
        else:
            # For training/validation, we return the labels
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
