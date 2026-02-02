import os
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(img_size=Config.IMG_SIZE, mode="train"):
    """
    Returns the data augmentation pipeline using Albumentations.

    Args:
        img_size (int): The target image size (height and width).
        mode (str): 'train' for training augmentations, 'valid' or 'test' for inference.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                # Resize to slightly larger than target to allow for random cropping
                # This is a mild augmentation strategy suitable for fine-grained classification
                A.Resize(height=int(img_size * 1.1), width=int(img_size * 1.1)),
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic transform for validation and testing
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


def apk(actual, predicted, k=5):
    """
    Computes the Average Precision at k (AP@k) for a single sample.

    Args:
        actual (list): A list of ground truth items (e.g., [correct_id]).
        predicted (list): A list of predicted items (ordered by confidence).
        k (int): The maximum number of predicted elements to consider.

    Returns:
        float: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the Mean Average Precision at k (MAP@k) across all samples.

    Args:
        actual (list of lists): A list where each element is a list of ground truth items.
        predicted (list of lists): A list where each element is a list of predicted items.
        k (int): The maximum number of predicted elements to consider.

    Returns:
        float: The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
