import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(mode="train", img_size=384):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.

    Returns:
        A.Compose: Composed transforms.
    """
    # Standard ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # CoarseDropout (Cutout) as primary regularization
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles loading images and multi-hot encoding of labels.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image, labels, file_path).
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.class_labels = Config.class_labels
        self.num_classes = len(self.class_labels)

        # Create a mapping from label string to index
        self.label_to_idx = {label: i for i, label in enumerate(self.class_labels)}

        # Pre-process labels for training and validation
        if self.mode != "test":
            self.labels = self._process_labels()
        else:
            # Dummy labels for test set
            self.labels = np.zeros((len(self.df), self.num_classes), dtype=np.float32)

    def _process_labels(self):
        """
        Converts space-delimited label strings into multi-hot encoded numpy arrays.
        """
        encoded_labels = np.zeros((len(self.df), self.num_classes), dtype=np.float32)

        for idx, row in self.df.iterrows():
            label_str = row["labels"]
            if not isinstance(label_str, str):
                continue

            current_labels = label_str.split()
            for label in current_labels:
                if label in self.label_to_idx:
                    class_idx = self.label_to_idx[label]
                    encoded_labels[idx, class_idx] = 1.0

        return encoded_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # Metadata file_path is relative to Config.input_dir (e.g., "train_images/abc.jpg")
        rel_path = self.df.loc[idx, "file_path"]
        full_path = os.path.join(Config.input_dir, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing/corrupt images (should be caught by metadata check, but for safety)
            # Create a black image
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
