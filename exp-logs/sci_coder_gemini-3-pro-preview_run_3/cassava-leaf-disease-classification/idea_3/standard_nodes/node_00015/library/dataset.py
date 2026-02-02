import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


def get_transforms(data="train"):
    """
    Returns the Albumentations transformations for training or validation/testing.

    Args:
        data (str): 'train' for training augmentations, 'valid' or 'test' for validation/inference.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.25
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(self, df, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transform (A.Compose, optional): Albumentations transforms to apply.
            output_label (bool): Whether to return the label (True for train/val, False for pure inference if labels missing).
        """
        self.df = df
        self.file_paths = df["file_path"].values
        # Handle labels if they exist and are requested
        if output_label:
            self.labels = df["label"].values
        else:
            self.labels = None
        self.transform = transform
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # metadata file_path is relative to input_root (e.g., "train_images/123.jpg")
        # CFG.input_root is "./input"
        file_path = os.path.join(CFG.input_root, self.file_paths[idx])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label if available
        if self.output_label:
            label = torch.tensor(self.labels[idx]).long()
            return image, label
        else:
            return image


class Mixup:
    """
    Implements Mixup and CutMix augmentation.
    """

    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, num_classes=5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def rand_bbox(self, size, lam):
        """
        Generates a random bounding box for CutMix.
        """
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

    def __call__(self, batch, target):
        """
        Applies Mixup or CutMix to the batch.

        Args:
            batch (torch.Tensor): Batch of images [B, C, H, W]
            target (torch.Tensor): Batch of labels [B]

        Returns:
            torch.Tensor: Mixed images
            torch.Tensor: Mixed targets (soft labels) [B, num_classes]
        """
        # Check if we should apply mixing
        if np.random.rand() > self.prob:
            # Return one-hot encoded targets if no mixup applied, for consistency in loss calculation
            return (
                batch,
                torch.nn.functional.one_hot(
                    target, num_classes=self.num_classes
                ).float(),
            )

        # Decide between Mixup and CutMix
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            alpha = self.cutmix_alpha
        else:
            alpha = self.mixup_alpha

        lam = np.random.beta(alpha, alpha)
        batch_size = batch.size(0)
        rand_index = torch.randperm(batch_size).to(batch.device)

        target_a = target
        target_b = target[rand_index]

        # Create one-hot targets
        target_a_onehot = torch.nn.functional.one_hot(
            target_a, num_classes=self.num_classes
        ).float()
        target_b_onehot = torch.nn.functional.one_hot(
            target_b, num_classes=self.num_classes
        ).float()

        if use_cutmix:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)

            # Adjust lambda to match pixel ratio exactly
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2])
            )

            batch[:, :, bby1:bby2, bbx1:bbx2] = batch[
                rand_index, :, bby1:bby2, bbx1:bbx2
            ]

            # Mix targets
            mixed_target = lam * target_a_onehot + (1 - lam) * target_b_onehot

        else:
            # Mixup
            mixed_batch = lam * batch + (1 - lam) * batch[rand_index, :]
            mixed_target = lam * target_a_onehot + (1 - lam) * target_b_onehot
            batch = mixed_batch

        return batch, mixed_target
