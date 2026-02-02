import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the transformation pipeline for the specified phase.

    Args:
        phase (str): One of 'train', 'val', 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_list = []

    # OpenCV loads images as numpy arrays (H, W, C).
    # Convert to PIL Image to use torchvision's Resize and other transforms efficiently.
    transform_list.append(transforms.ToPILImage())

    # Resize to the fixed input size expected by the model (224x224)
    transform_list.append(transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)))

    if phase == "train":
        # Data augmentation: Randomly flip horizontally
        # Laterality (left/right eye) does not dictate severity, making this safe.
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Convert PIL Image to Tensor (C, H, W) in range [0, 1]
    transform_list.append(transforms.ToTensor())

    # Normalize using ImageNet mean and std
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for loading Diabetic Retinopathy images and labels.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing at least 'file_path' and optionally 'diagnosis'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

        # Ensure file_path column exists
        if "file_path" not in self.df.columns:
            raise ValueError("DataFrame must contain 'file_path' column.")

        # Extract file paths
        self.file_paths = self.df["file_path"].values

        # Extract labels if available, otherwise use dummy labels (for inference)
        # We cast to float32 because we are doing regression
        if "diagnosis" in self.df.columns:
            self.labels = self.df["diagnosis"].values.astype(np.float32)
        else:
            self.labels = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads and processes a single sample.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            tuple: (image_tensor, label_float)
        """
        # Construct full image path
        # Metadata paths are relative (e.g., 'train_images/id.png'), so we join with input root
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Safety check for missing or corrupt images
        if image is None:
            # Return a blank image to avoid crashing the training loop
            # This should not happen given the data verification steps
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformation pipeline
        if self.transform:
            image = self.transform(image)

        # Get the label as a float tensor for regression
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return image, label
