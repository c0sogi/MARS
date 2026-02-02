import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data, cfg):
    """
    Returns the Albumentations transformations for the specified data split.

    Args:
        data (str): One of 'train', 'valid', 'test'.
        cfg (Config): Configuration object.

    Returns:
        A.Compose: Composed transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Valid and Test
        return A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Supports Standard training, Distillation, and Testing modes.
    """

    def __init__(self, df, cfg, transform=None, mode="standard", teacher_logits=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, targets).
            cfg (Config): Configuration object.
            transform (albumentations.Compose): Transformations to apply.
            mode (str): 'standard', 'distillation', or 'test'.
            teacher_logits (dict): Dictionary mapping image_id to logits (numpy array).
                                   Required for 'distillation' mode.
        """
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.teacher_logits = teacher_logits

        self.image_ids = self.df["image_id"].values
        self.file_paths = self.df["file_path"].values

        # Prepare labels for training/validation
        if self.mode in ["standard", "distillation"]:
            # Ensure target columns exist
            if not all(col in self.df.columns for col in self.cfg.target_cols):
                raise ValueError(
                    f"Target columns {self.cfg.target_cols} not found in DataFrame."
                )

            # Convert one-hot/probabilistic labels to class indices for CrossEntropy
            # We use argmax to get the dominant class index (0, 1, 2, 3)
            self.labels = self.df[self.cfg.target_cols].values.argmax(axis=1)

        if self.mode == "distillation" and self.teacher_logits is None:
            raise ValueError(
                "teacher_logits dictionary must be provided for distillation mode."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve image path
        filename = self.file_paths[idx]
        img_path = self.cfg.get_image_path(filename)

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        img_id = self.image_ids[idx]

        # Return based on mode
        if self.mode == "test":
            return image, img_id

        # Get Ground Truth Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.mode == "distillation":
            # Retrieve Teacher Logits
            # Expecting teacher_logits to be a dict: {image_id: np.array([logit_0, ...])}
            t_logits = self.teacher_logits.get(img_id)
            if t_logits is None:
                raise KeyError(
                    f"Logits for image {img_id} not found in teacher_logits."
                )

            t_logits = torch.tensor(t_logits, dtype=torch.float32)
            return image, label, t_logits

        # Standard Mode
        return image, label
