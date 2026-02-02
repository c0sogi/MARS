import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformations for the specified data mode.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
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
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def load_data(mode, load_cached_data=True):
    """
    Loads metadata for the given mode. Implements caching using Parquet.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    cache_path = os.path.join(Config.output_dir, f"{mode}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reloading from source.")

    # 2. Load from source metadata
    if mode == "train":
        source_path = Config.train_metadata_path
    elif mode == "val":
        source_path = Config.val_metadata_path
    elif mode == "test":
        source_path = Config.test_metadata_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pd.read_csv(source_path, dtype={"id": str, "attribute_ids": str})

    # Process paths
    # The metadata file_path is relative to ./input (e.g. "train/xxx.png")
    # We construct the full absolute path or path relative to current working dir
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.input_dir, x))

    # Handle NaNs in attribute_ids for test set or missing labels
    if "attribute_ids" not in df.columns:
        df["attribute_ids"] = ""
    else:
        df["attribute_ids"] = df["attribute_ids"].fillna("")

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class ArtworkDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.num_classes = Config.num_classes

        # Pre-process labels for faster access during training
        self.labels = []
        if self.mode != "test":
            for attr_str in self.df["attribute_ids"]:
                if not attr_str.strip():
                    self.labels.append([])
                else:
                    self.labels.append([int(x) for x in attr_str.split()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]
        image_id = row["id"]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (though verification script showed 0 missing)
            # Create a black image to prevent crashing
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Prepare target
        if self.mode == "test":
            # Dummy target for test
            target = torch.zeros(self.num_classes, dtype=torch.float32)
        else:
            # Multi-hot encoding
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            indices = self.labels[idx]
            if indices:
                target[indices] = 1.0

        return image, target, image_id


class MixupCutMix:
    """
    Applies Mixup or CutMix to a batch of images and labels.
    Designed to be called within the training loop or as a collate function wrapper.
    """

    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5, num_classes=3474):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

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

    def __call__(self, images, targets):
        """
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
            targets (torch.Tensor): Batch of multi-hot labels (B, NumClasses)

        Returns:
            mixed_images, mixed_targets
        """
        # Decide whether to apply augmentation
        if np.random.rand() > self.prob:
            return images, targets

        # Decide between Mixup and CutMix
        # We'll use a 50/50 split if the augmentation is triggered
        use_cutmix = np.random.rand() > 0.5

        batch_size = images.size(0)
        rand_index = torch.randperm(batch_size).to(images.device)

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

            # Adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
            )

            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
                rand_index, :, bby1:bby2, bbx1:bbx2
            ]

            # Mix targets
            target_a = targets
            target_b = targets[rand_index]
            mixed_targets = target_a * lam + target_b * (1.0 - lam)

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            mixed_images = lam * images + (1 - lam) * images[rand_index]

            # Mix targets
            target_a = targets
            target_b = targets[rand_index]
            mixed_targets = target_a * lam + target_b * (1.0 - lam)

        return mixed_images, mixed_targets


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load DataFrames
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Debug mode: sample small subset
    if Config.debug:
        train_df = train_df.iloc[: Config.debug_sample_size]
        val_df = val_df.iloc[: Config.debug_sample_size]
        # We generally want to predict on full test even in debug to ensure pipeline works,
        # but for speed we can sample.
        test_df = test_df.iloc[: Config.debug_sample_size]

    # Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = ArtworkDataset(val_df, transforms=get_transforms("valid"), mode="val")

    test_dataset = ArtworkDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,  # Useful for batch norm and mixup stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
