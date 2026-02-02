import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything


def get_transforms(mode, cfg):
    """
    Returns the torchvision transforms for the specified mode.
    Ensures PIL compatibility and implements the proposed augmentation pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # RandomResizedCrop is critical for preventing overfitting on global patterns
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                # RandAugment provides diverse photometric distortions
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test: Deterministic processing
        # Resize slightly larger then crop to target size is standard practice
        # Maintain crop ratio of 0.875 (224/256)
        crop_size = cfg.image_size
        resize_size = int(crop_size / 0.875)
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class CassavaDataset(Dataset):
    def __init__(self, df, transform=None, mode="train", input_dir=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
            input_dir (str): Base directory for images.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = input_dir if input_dir else Config.input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path using the relative path from metadata
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using PIL (Strictly enforced)
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            # Fallback for potential read errors to prevent training crash
            # This creates a black image of the correct size
            print(
                f"Warning: Could not load {file_path}, generating black image. Error: {e}"
            )
            image = Image.new("RGB", (Config.image_size, Config.image_size), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.mode == "test":
            # For inference, we need the ID to map predictions back to files
            return image, row["image_id"]
        else:
            # For training/validation, we need the label
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.long)


class MixupCutmixCollator:
    """
    Custom Collator that applies MixUp or CutMix regularization to a batch.
    Also handles Label Smoothing for consistency.
    """

    def __init__(
        self,
        num_classes,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        label_smoothing=0.1,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing

    def rand_bbox(self, size, lam):
        """Generates a random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)

        batch_size = imgs.size(0)

        # Initialize targets with Label Smoothing
        # Formula: (1 - epsilon) * one_hot + epsilon / K
        # This ensures that even unmixed batches return soft targets compatible with SoftTargetCrossEntropy
        targets = torch.zeros(batch_size, self.num_classes, device=imgs.device)
        targets.scatter_(1, labels.view(-1, 1), 1)
        targets = (
            targets * (1 - self.label_smoothing)
            + self.label_smoothing / self.num_classes
        )

        # Probabilistically apply MixUp or CutMix
        if np.random.rand() < self.prob:
            # Decide MixUp vs CutMix (50/50 split)
            use_cutmix = np.random.rand() < 0.5

            # Generate permutation for mixing
            indices = torch.randperm(batch_size)
            shuffled_imgs = imgs[indices]
            shuffled_targets = targets[indices]

            if use_cutmix:
                # CutMix Implementation
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)

                # Adjust lambda to match the exact pixel area removed
                lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))
                )

                # Apply patch
                imgs[:, :, bbx1:bbx2, bby1:bby2] = shuffled_imgs[
                    :, :, bbx1:bbx2, bby1:bby2
                ]
                # Mix targets
                targets = lam * targets + (1 - lam) * shuffled_targets

            else:
                # MixUp Implementation
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                # Linear interpolation of images
                imgs = lam * imgs + (1 - lam) * shuffled_imgs
                # Linear interpolation of targets
                targets = lam * targets + (1 - lam) * shuffled_targets

        return imgs, targets


def get_dataloaders(cfg):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    seed_everything(cfg.seed)

    # Load Metadata CSVs
    train_df = pd.read_csv(cfg.train_metadata_path)
    val_df = pd.read_csv(cfg.val_metadata_path)
    test_df = pd.read_csv(cfg.test_metadata_path)

    # Handle Debug Mode (Subsampling)
    if cfg.debug:
        print(f"Debug mode enabled. Subsampling {cfg.debug_sample_size} samples.")
        train_df = train_df.head(cfg.debug_sample_size)
        val_df = val_df.head(cfg.debug_sample_size)
        test_df = test_df.head(cfg.debug_sample_size)

    # Initialize Transforms
    train_transform = get_transforms("train", cfg)
    val_transform = get_transforms(
        "val", cfg
    )  # Validation and Test use the same deterministic transform

    # Initialize Datasets
    train_dataset = CassavaDataset(train_df, transform=train_transform, mode="train")
    val_dataset = CassavaDataset(val_df, transform=val_transform, mode="val")
    test_dataset = CassavaDataset(test_df, transform=val_transform, mode="test")

    # Initialize Collator for Training
    # This handles the MixUp/CutMix logic and label smoothing
    train_collator = MixupCutmixCollator(
        num_classes=cfg.num_classes,
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        prob=cfg.mixup_prob,
        label_smoothing=cfg.label_smoothing,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=train_collator,  # Use custom collator
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain consistent batch statistics
    )

    # Validation Loader
    # Uses standard collate (default) as we don't mix validation data
    # Note: Validation loop should expect (image, label_index) tuples, unlike train loop which gets (image, soft_targets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
