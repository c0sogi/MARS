import os
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(logits, targets):
    """
    Computes the Mean F1-Score (Macro) from logits and targets.
    Applies Sigmoid activation and a threshold of 0.5 to logits.

    Args:
        logits (torch.Tensor): Raw model outputs of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).

    Returns:
        float: The macro-averaged F1 score.
    """
    # Move tensors to CPU and detach from computation graph
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()

    # Apply Sigmoid activation to convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions using the threshold
    preds = (probs > 0.5).int().numpy()
    targets = targets.int().numpy()

    # Calculate Macro F1 Score
    # zero_division=0 ensures that if a class is missing in the batch, f1 is 0 for that class
    # instead of raising an error or warning.
    score = f1_score(targets, preds, average="macro", zero_division=0)

    return score


def get_transforms(data_type="train"):
    """
    Returns the Albumentations composition for data transformation.

    Args:
        data_type (str): 'train' for training transforms (with augmentation),
                         'valid' or 'test' for deterministic transforms.

    Returns:
        A.Compose: The composition of transforms.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data_type == "train":
        return A.Compose(
            [
                # Cite solution_lesson_node_00004: Prefer RandomResizedCrop over deterministic resizing for better regularization
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=(0.5, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Cite solution_lesson_node_00001: Add ColorJitter to fix generalization gap and lighting sensitivity
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation or Test transforms (No augmentation)
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
