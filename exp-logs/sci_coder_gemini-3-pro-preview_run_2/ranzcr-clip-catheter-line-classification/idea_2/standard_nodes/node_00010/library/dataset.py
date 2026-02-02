import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Strategy:
    1. Resize (Longest Edge) -> Maintains Aspect Ratio
    2. CLAHE -> Enhance local contrast before geometric transforms (Cite solution_lesson_node_00006)
                and before padding (Cite solution_lesson_node_00007).
    3. Geometric Augmentations (Train only) -> ShiftScaleRotate
    4. Pad -> Square (Zero padding)
    5. Normalize -> ImageNet stats
    6. ToTensor
    """
    transforms = [
        # Resize such that the longest side is equal to IMAGE_SIZE, maintaining aspect ratio
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
        # Apply CLAHE to enhance contrast of catheters/lines
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ]

    if mode == "train":
        transforms.extend(
            [
                # Geometric augmentations to prevent overfitting
                # Applied after CLAHE (Cite solution_lesson_node_00006) but before Padding
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
            ]
        )

    transforms.extend(
        [
            # Pad the shorter side with zeros (black) to make it square
            A.PadIfNeeded(
                min_height=Config.IMAGE_SIZE,
                min_width=Config.IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            # Normalize using ImageNet mean and std
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # Convert to PyTorch Tensor
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter and Line Position Detection.
    Reads images based on metadata paths and applies the specified preprocessing pipeline.
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.mode = mode
        self.metadata_path = metadata_path

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle Debug Mode
        if Config.DEBUG:
            # Deterministic sampling for debug
            sample_size = min(len(self.df), Config.DEBUG_SAMPLE_SIZE)
            self.df = self.df.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

        # Setup Transforms
        self.transform = transform
        if self.transform is None:
            self.transform = get_transforms(mode)

        self.target_cols = Config.TARGET_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # metadata contains relative path (e.g., "train/1.2.3.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)

        # Robustness check for missing/corrupt images
        if image is None:
            # Return a black image of appropriate size to prevent crash
            # We start with a placeholder that will be resized and padded.
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Preprocessing Pipeline
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return (Image, Label)
        if self.mode in ["train", "val"]:
            # Extract targets
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # For test/inference, return dummy labels
            # This maintains a consistent signature for the DataLoader
            num_classes = len(self.target_cols)
            return image, torch.zeros(num_classes, dtype=torch.float32)
