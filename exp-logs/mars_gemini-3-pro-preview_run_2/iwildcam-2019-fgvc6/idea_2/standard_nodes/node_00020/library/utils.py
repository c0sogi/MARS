import os
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(phase: str):
    """
    Returns the data transformations for the specified phase.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composed transformations.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif phase in ["val", "test"]:
        return transforms.Compose(
            [
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'train', 'val', or 'test'.")


def compute_class_weights(df: pd.DataFrame):
    """
    Computes class weights to handle class imbalance using the inverse frequency method.
    Formula: weight[c] = total_samples / (num_classes * count[c])

    Args:
        df (pd.DataFrame): The training metadata containing the 'Category' column.

    Returns:
        torch.FloatTensor: A tensor of shape (num_classes,) containing the weights.
    """
    if "Category" not in df.columns:
        raise ValueError("DataFrame must contain a 'Category' column.")

    # Get all labels
    labels = df["Category"].values

    # Use the number of classes defined in Config
    num_classes = Config.NUM_CLASSES

    # Calculate counts for each class ID (0 to num_classes-1)
    class_counts = np.bincount(labels, minlength=num_classes)

    total_samples = len(labels)

    # Initialize weights array
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        count = class_counts[c]
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            # Handle cases where a class might be missing in the split (unlikely but possible)
            # Set weight to 1.0 to avoid division by zero or NaN
            weights[c] = 1.0

    return torch.FloatTensor(weights)
