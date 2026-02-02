import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


def get_transforms(data: str):
    """
    Returns the Albumentations transform pipeline for a given data split.

    Args:
        data (str): 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                # Restrict scale to preserve context (0.3 instead of 0.08)
                A.RandomResizedCrop(
                    size=(CFG.image_size, CFG.image_size), scale=(0.3, 1.0)
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(height=CFG.image_size, width=CFG.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Test time transforms (same as valid usually, or with TTA logic handled elsewhere)
        return A.Compose(
            [
                A.Resize(height=CFG.image_size, width=CFG.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Reads images based on metadata paths and applies transforms.
    """

    def __init__(self, df, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'file_path' and 'label'.
            transform (albumentations.Compose): Transforms to apply.
            output_label (bool): Whether to return the label (True for train/val, False for test).
        """
        self.df = df
        self.transform = transform
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve row
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train_images/1000015157.jpg"
        # CFG.input_root is "./input"
        file_path = os.path.join(CFG.input_root, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label (if requested)
        if self.output_label:
            label = row["label"]
            return image, label
        else:
            return image


class CollateMixupCutmix:
    """
    Collate function that applies MixUp or CutMix to a batch of images and labels.
    Converts integer labels to one-hot encoded soft targets.
    """

    def __init__(self, mix_p=1.0, alpha=0.4, n_classes=5, label_smoothing=0.0):
        self.mix_p = mix_p
        self.alpha = alpha
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, label)
        Returns:
            images: Tensor of shape (B, C, H, W)
            targets: Tensor of shape (B, n_classes) - Soft targets
        """
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        # Convert labels to one-hot (with optional label smoothing)
        batch_size = images.size(0)
        one_hot_targets = torch.zeros(batch_size, self.n_classes).to(images.device)

        if self.label_smoothing > 0:
            # Apply label smoothing: (1 - epsilon) * y + epsilon / K
            one_hot_targets.fill_(self.label_smoothing / self.n_classes)
            one_hot_targets.scatter_(
                1,
                labels.view(-1, 1),
                1.0 - self.label_smoothing + (self.label_smoothing / self.n_classes),
            )
        else:
            one_hot_targets.scatter_(1, labels.view(-1, 1), 1)

        # Decide whether to apply mixing
        if np.random.rand() > self.mix_p:
            return images, one_hot_targets

        # Generate mixing parameters
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)

        # Decide MixUp vs CutMix (50/50 probability)
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

            # Adjust lambda to exact area ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
            )

            # Apply CutMix to images
            images[:, :, bby1:bby2, bbx1:bbx2] = images[
                rand_index, :, bby1:bby2, bbx1:bbx2
            ]

            # Mix targets
            targets = lam * one_hot_targets + (1 - lam) * one_hot_targets[rand_index]
        else:
            # MixUp
            images = lam * images + (1 - lam) * images[rand_index]
            targets = lam * one_hot_targets + (1 - lam) * one_hot_targets[rand_index]

        return images, targets


def get_dataframe(split, load_cached_data=True):
    """
    Loads the dataframe for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Placeholder for consistency, though we read CSVs directly.
    """
    # Ensure cache directory exists (as per requirement, though we rely on ./metadata)
    os.makedirs(CFG.output_dir, exist_ok=True)

    if split == "train":
        return pd.read_csv(CFG.train_csv)
    elif split == "val":
        return pd.read_csv(CFG.val_csv)
    elif split == "test":
        return pd.read_csv(CFG.test_csv)
    else:
        raise ValueError(f"Unknown split: {split}")
