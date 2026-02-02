import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    size: (N, C, H, W)
    lam: lambda value derived from Beta distribution
    """
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample center of the box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Mixup:
    """
    Applies Mixup or CutMix to a batch of images and labels.
    Returns mixed images and soft targets.
    """

    def __init__(self, mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0, num_classes=5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, batch, target):
        """
        Args:
            batch: Tensor of shape (N, C, H, W)
            target: Tensor of shape (N) (LongTensor)
        Returns:
            batch: Mixed images
            target: Mixed soft labels (N, C) or original labels if no mixup
        """
        # If probability check fails or mixup is disabled (prob <= 0), return original
        if self.prob <= 0 or np.random.rand() > self.prob:
            # Convert to one-hot to ensure compatibility with SoftTargetCrossEntropy
            target_onehot = torch.zeros(
                batch.size(0), self.num_classes, device=batch.device
            )
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            return batch, target_onehot

        # Convert target to one-hot
        target_onehot = torch.zeros(
            batch.size(0), self.num_classes, device=batch.device
        )
        target_onehot.scatter_(1, target.view(-1, 1), 1)

        # Decide between Mixup and CutMix
        # If both are enabled, we choose with 50% probability
        use_cutmix = (self.cutmix_alpha > 0) and (np.random.rand() < 0.5)

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(batch.size(0)).to(batch.device)

            bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(), lam)

            # Adjust lambda to match the exact pixel ratio of the crop
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2])
            )

            batch[:, :, bbx1:bbx2, bby1:bby2] = batch[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            target_mixed = lam * target_onehot + (1 - lam) * target_onehot[rand_index]

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(batch.size(0)).to(batch.device)

            batch = lam * batch + (1 - lam) * batch[rand_index]
            target_mixed = lam * target_onehot + (1 - lam) * target_onehot[rand_index]

        return batch, target_mixed


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Uses PIL for image loading.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # file_path is relative to input dir (e.g., "train_images/123.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Open image and convert to RGB (handles RGBA or Grayscale)
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            # Fallback for read errors (unlikely given EDA)
            print(f"Error loading image {full_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Label is an integer
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


def get_transforms(phase, image_size):
    """
    Returns the augmentation pipeline for the specified phase.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                # Geometric Augmentation: Scale invariance
                transforms.RandomResizedCrop(image_size),
                # Geometric Augmentation: Orientation invariance
                transforms.RandomHorizontalFlip(),
                # Photometric Augmentation: Robustness to lighting/color
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif phase == "val" or phase == "test":
        return transforms.Compose(
            [
                # Resize shorter edge to image_size
                transforms.Resize(image_size),
                # Center crop to square
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


def get_dataloaders(train_df, val_df, test_df, phase_config):
    """
    Creates DataLoaders for train, validation, and test sets based on phase config.
    """
    image_size = phase_config["image_size"]
    batch_size = phase_config["batch_size"]
    num_workers = Config.NUM_WORKERS

    # Get transforms
    train_transform = get_transforms("train", image_size)
    val_transform = get_transforms("val", image_size)

    # Initialize Datasets
    train_ds = CassavaDataset(train_df, transform=train_transform)
    val_ds = CassavaDataset(val_df, transform=val_transform)
    test_ds = CassavaDataset(test_df, transform=val_transform)

    # Initialize DataLoaders
    train_loader = None
    if len(train_ds) > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    test_loader = None
    if len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
