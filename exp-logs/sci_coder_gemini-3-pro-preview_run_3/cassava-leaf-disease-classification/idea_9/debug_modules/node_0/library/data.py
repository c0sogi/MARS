import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Loads images based on metadata paths and applies augmentations.
    """

    def __init__(self, metadata_path, transform=None, is_train=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Augmentation pipeline.
            is_train (bool): Whether the dataset is for training (affects label handling).
        """
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.is_train = is_train
        self.root_dir = Config.input_root

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Construct full file path
        # metadata file_path is relative to input_root (e.g., "train_images/123.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle label
        if self.is_train:
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            # For test set, return dummy label or just image
            # Returning dummy label 0 to keep signature consistent
            return image, torch.tensor(0, dtype=torch.long)


def get_transforms(data="train", img_size=Config.img_size):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' or 'valid'.
        img_size (int): Target image size.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Geometric Augmentations (D4 Group + Affine)
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                # Normalization and Tensor conversion
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
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_cutmix_fn(images, labels, alpha=1.0, prob=0.5):
    """
    Applies MixUp or CutMix to a batch of images and labels.

    Args:
        images (torch.Tensor): Batch of images [B, C, H, W].
        labels (torch.Tensor): Batch of labels [B].
        alpha (float): Beta distribution parameter.
        prob (float): Probability of applying the augmentation (vs doing nothing?
                      Usually this prob decides between Mixup vs Cutmix if both active,
                      or if we apply it at all. Here we assume we apply one of them).

    Returns:
        mixed_images, target_a, target_b, lam
    """
    # Decide whether to apply Mixup/Cutmix or not?
    # Usually in training loops, this function is called if we WANT to apply it.
    # We will randomly choose between MixUp and CutMix.

    batch_size = images.size(0)
    indices = torch.randperm(batch_size).to(images.device)

    target_a = labels
    target_b = labels[indices]

    # Generate lambda from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Decide between MixUp and CutMix
    # We use 'prob' here as the threshold for CutMix vs MixUp
    # e.g., if random < 0.5 -> MixUp, else -> CutMix
    if np.random.rand() < 0.5:
        # MixUp
        mixed_images = lam * images + (1 - lam) * images[indices, :]
    else:
        # CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[
            indices, :, bbx1:bbx2, bby1:bby2
        ]
        # Adjust lambda to match pixel ratio exactly
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
        )

    return mixed_images, target_a, target_b, lam
