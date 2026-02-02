import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for a given split.
    """
    mean = Config.MEAN
    std = Config.STD

    transforms_list = []

    # Resize to ensure consistency (dataset is 32x32)
    transforms_list.append(A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE))

    if split == "train":
        # Geometric Augmentations
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.VerticalFlip(p=0.5))

    # Normalization and Tensor conversion
    # Albumentations Normalize uses max_pixel_value=255.0 by default,
    # so it handles the 0-255 -> 0-1 scaling internally before standardizing.
    transforms_list.append(
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True)
    )
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class CactusDataset(Dataset):
    """
    Dataset class for loading Cactus images.
    """

    def __init__(self, metadata_path, transform=None, is_test=False):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        if Config.DEBUG:
            return min(len(self.df), Config.DEBUG_SUBSET_SIZE)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen based on metadata check)
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.is_test:
            return image, row["id"]
        else:
            label = torch.tensor(row["has_cactus"], dtype=torch.float32)
            return image, label


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    seed_everything(Config.SEED)

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    # Datasets
    train_dataset = CactusDataset(
        Config.TRAIN_METADATA_PATH, transform=train_transform, is_test=False
    )
    val_dataset = CactusDataset(
        Config.VAL_METADATA_PATH, transform=val_transform, is_test=False
    )
    test_dataset = CactusDataset(
        Config.TEST_METADATA_PATH, transform=test_transform, is_test=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
