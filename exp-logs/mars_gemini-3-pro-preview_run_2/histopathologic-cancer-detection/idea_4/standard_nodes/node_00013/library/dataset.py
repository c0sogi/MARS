import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for loading Pathology images.
    Reads images via OpenCV, converts to RGB, and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing 'id', 'file_path', and optionally 'label'.
            transforms (albumentations.Compose, optional): Transformations to apply to the image.
        """
        self.df = df
        self.transforms = transforms

        # Pre-fetch paths and labels for efficient access
        self.file_paths = df["file_path"].values
        self.ids = df["id"].values

        if "label" in df.columns:
            self.labels = df["label"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.input_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return image and label
        if self.labels is not None:
            label = self.labels[idx]
            # BCEWithLogitsLoss requires float targets
            target = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return image, target
        else:
            # Return dummy target for test set
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_transforms(data="train"):
    """
    Creates the Albumentations transform pipeline based on the data split.
    Implements the 'Augment-then-Crop' strategy.

    Args:
        data (str): 'train' or 'valid' (also used for test).

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    mean = Config.dataset_mean
    std = Config.dataset_std
    crop_size = Config.crop_size

    transforms_list = []

    if data == "train":
        # 1. Global Augmentations on the full source patch (96x96)
        # This ensures rotation/color stats use the full context
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
            ]
        )

    # 2. Contextual Crop
    # Crop the center 64x64 region (ROI + context buffer)
    transforms_list.append(A.CenterCrop(height=crop_size, width=crop_size))

    # 3. Normalization and Tensor Conversion
    transforms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])

    return A.Compose(transforms_list)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Performs Mixup augmentation on a batch of data.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup beta distribution parameter.
        use_cuda (bool): Whether to use CUDA for index generation (deprecated, uses x.device).

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = x.device
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.

    Args:
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        pred: Model predictions.
        y_a: Original labels.
        y_b: Permuted labels.
        lam: Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
