import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformation pipeline based on the data type.

    Args:
        data_type (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Strategic Retention: Explicitly include VerticalFlip and HorizontalFlip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Robust Augmentation: ShiftScaleRotate and RandomBrightnessContrast
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    elif data_type in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, debug=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, targets).
            transforms (A.Compose, optional): Albumentations transformations.
            debug (bool): If True, subsets the data for debugging purposes.
        """
        self.df = df
        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        self.transforms = transforms

        # Create a mapping from class label to index
        self.label_map = {label: i for i, label in enumerate(Config.CLASS_LABELS)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to ./input (e.g., "images/Train_0.jpg")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            # (Though get_transforms should usually be used)
            transform = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = transform(image=image)["image"]

        # Handle Targets
        # If 'stratify_label' or target columns exist, return label index
        # Otherwise (test set), return a dummy label (-1)

        label_index = -1

        if "stratify_label" in row:
            label_name = row["stratify_label"]
            label_index = self.label_map.get(label_name, -1)
        else:
            # Try to determine label from one-hot columns if stratify_label is missing
            # This handles cases where we might be using a df without stratify_label but with targets
            for i, class_name in enumerate(Config.CLASS_LABELS):
                if class_name in row and row[class_name] == 1:
                    label_index = i
                    break

        # Return image and label (as long tensor for CrossEntropy)
        return image, torch.tensor(label_index, dtype=torch.long)
