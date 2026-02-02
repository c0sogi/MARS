import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    scale=(Config.AUG_MIN_SCALE, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


class DogCatDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path. Metadata filepath is relative to Config.INPUT_DIR
        # e.g., "train/cat.0.jpg" -> "./input/train/cat.0.jpg"
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing/corrupt images gracefully by returning a blank image or raising error
            # For this competition context, we assume data integrity or raise error
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            # Label is 0 or 1. Convert to float for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # Test mode: return image and id
            img_id = row["id"]
            return image, img_id


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    """
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


class MixupCutmixCollate:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, batch):
        """
        Applies Mixup or CutMix to the batch.
        Batch is a list of tuples (image, label).
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        # Decide whether to apply augmentation
        if np.random.rand() > self.prob:
            return images, labels

        # Decide between Mixup and CutMix (50/50)
        use_cutmix = np.random.rand() > 0.5

        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

            # Adjust lambda to match exact pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
            )

            images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[
                :, :, bbx1:bbx2, bby1:bby2
            ]
            targets = lam * labels + (1.0 - lam) * shuffled_labels

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = lam * images + (1.0 - lam) * shuffled_images
            targets = lam * labels + (1.0 - lam) * shuffled_labels

        return images, targets


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int, optional): Override Config.BATCH_SIZE.
        num_workers (int, optional): Override Config.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Create Datasets
    train_dataset = DogCatDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = DogCatDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_dataset = DogCatDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # Initialize Collate Function for Training
    mixup_collate = MixupCutmixCollate(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.PROB_AUG,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mixup_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
