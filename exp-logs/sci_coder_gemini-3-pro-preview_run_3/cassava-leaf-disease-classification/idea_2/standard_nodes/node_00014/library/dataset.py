import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import CFG


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    Reads images via OpenCV, converts to RGB, and applies Albumentations transforms.
    """

    def __init__(self, df, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
            transform (albumentations.Compose, optional): Transformation pipeline.
            output_label (bool): If True, returns (image, label). If False, returns (image).
        """
        self.df = df
        self.transform = transform
        self.output_label = output_label

        # Cache paths and labels to numpy arrays for faster access during indexing
        self.file_paths = self.df["file_path"].values
        if self.output_label:
            self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Construct full file path
        # Metadata contains relative paths (e.g., "train_images/123.jpg")
        # CFG.input_root is "./input"
        relative_path = self.file_paths[index]
        full_path = os.path.join(CFG.input_root, relative_path)

        # Load image
        img = cv2.imread(full_path)

        if img is None:
            raise FileNotFoundError(f"Could not load image at {full_path}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return data
        if self.output_label:
            label = self.labels[index]
            # Return label as a LongTensor for CrossEntropyLoss
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img
