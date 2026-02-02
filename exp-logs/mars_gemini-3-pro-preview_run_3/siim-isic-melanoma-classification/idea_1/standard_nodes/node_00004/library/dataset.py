import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data_split):
    """
    Returns the albumentations transforms for the specific data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    image_size = Config.IMAGE_SIZE

    # ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms_list = []

    # Resize is applied to all splits to ensure compatibility with the model
    transforms_list.append(A.Resize(height=image_size, width=image_size))

    if data_split == "train":
        # Augmentations for training
        # Horizontal flip is a safe, invariant augmentation for dermatoscopy
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.VerticalFlip(p=0.5))
        transforms_list.append(
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5)
        )
        transforms_list.append(A.RandomBrightnessContrast(p=0.5))
        # CoarseDropout for spatial regularization
        transforms_list.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=24,
                max_width=24,
                min_holes=1,
                fill_value=0,
                p=0.5,
            )
        )

    # Normalization and Tensor conversion
    transforms_list.append(A.Normalize(mean=mean, std=std))
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class ISICDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        PyTorch Dataset for ISIC Skin Lesion Classification.

        Args:
            df (pd.DataFrame): Dataframe containing metadata and file paths.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'. Controls output format.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Check if target column exists (expected for train/val)
        self.has_target = "target" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Construct full image path
        # metadata file_path is relative to input dir (e.g., "jpeg/train/ISIC_...jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Handle potential missing images gracefully (though metadata validation should prevent this)
        if image is None:
            # Create a blank image if file read fails
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback: simple tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Prepare output based on mode
        output = {"image": image, "image_name": row["image_name"]}

        if self.mode in ["train", "val"] and self.has_target:
            # Return target as float for BCEWithLogitsLoss
            target = row["target"]
            output["target"] = torch.tensor(target, dtype=torch.float)

        return output


def load_dataset_dataframe(csv_path, debug_size=None):
    """
    Loads the dataset dataframe from CSV.

    Args:
        csv_path (str): Path to the CSV file.
        debug_size (int, optional): If set, limits the dataframe to this many rows.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if debug_size is not None:
        # Use a subset for debugging/testing
        df = df.iloc[:debug_size].copy()

    return df
