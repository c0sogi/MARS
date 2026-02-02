import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from library import config, utils


def load_metadata(mode, use_pseudo_labels=False, load_cached_data=True):
    """
    Loads metadata for the specified mode. Handles caching and pseudo-label merging.

    Args:
        mode (str): 'train', 'val', or 'test'.
        use_pseudo_labels (bool): If True and mode is 'train', merges test data with pseudo-labels.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed metadata dataframe.
    """
    # Determine cache path
    if mode == "train":
        cache_path = config.TRAIN_CACHE_PATH
        if use_pseudo_labels:
            base, ext = os.path.splitext(cache_path)
            cache_path = f"{base}_pseudo{ext}"
    elif mode == "val":
        cache_path = config.VAL_CACHE_PATH
    else:
        cache_path = config.TEST_CACHE_PATH

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")

    # Load raw metadata
    if mode == "train":
        df = pd.read_csv(config.TRAIN_METADATA_PATH)

        # Merge pseudo-labels if requested
        if use_pseudo_labels:
            if os.path.exists(config.PSEUDO_LABELS_PATH):
                df_test = pd.read_csv(config.TEST_METADATA_PATH)
                df_pseudo = pd.read_parquet(config.PSEUDO_LABELS_PATH)

                # Ensure pseudo labels have correct columns
                label_cols = [f"species_{i}" for i in range(config.NUM_CLASSES)]

                # Update test dataframe with pseudo labels
                # We assume pseudo_labels.parquet has rec_id and species columns
                df_test_merged = df_test.drop(columns=label_cols, errors="ignore")
                df_test_merged = df_test_merged.merge(
                    df_pseudo, on="rec_id", how="inner"
                )

                # Concatenate train and pseudo-labeled test
                df = pd.concat([df, df_test_merged], axis=0, ignore_index=True)
            else:
                print(
                    f"Warning: Pseudo-labels requested but {config.PSEUDO_LABELS_PATH} not found."
                )

    elif mode == "val":
        df = pd.read_csv(config.VAL_METADATA_PATH)
    else:
        df = pd.read_csv(config.TEST_METADATA_PATH)

    # Cache the result
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class BirdDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.mode = mode
        self.transform = transform
        self.label_cols = [f"species_{i}" for i in range(config.NUM_CLASSES)]

        # Pre-compute paths to avoid joining strings in loop
        self.file_paths = [
            os.path.join(config.INPUT_DIR, p) for p in df["file_path"].values
        ]

        # Pre-compute labels
        if self.mode != "test" or set(self.label_cols).issubset(df.columns):
            self.labels = df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(df), config.NUM_CLASSES), dtype=np.float32)

        # Map wav paths to spectrogram paths
        # Spectrograms are in config.SPECTROGRAM_DIR and have .bmp extension
        self.img_paths = []
        for fp in self.file_paths:
            basename = os.path.basename(fp)
            bmp_name = os.path.splitext(basename)[0] + ".bmp"
            self.img_paths.append(os.path.join(config.SPECTROGRAM_DIR, bmp_name))

    def __len__(self):
        if config.DEBUG_MAX_SAMPLES is not None:
            return min(len(self.df), config.DEBUG_MAX_SAMPLES)
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # Load Image
        # Load as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing files (should not happen based on EDA)
            image = np.zeros((config.IMG_HEIGHT, 1246), dtype=np.uint8)

        # Channel Replication (1 -> 3)
        image = cv2.merge([image, image, image])

        # Normalize to 0-1 before augmentations if needed, but Albumentations Normalize handles it.
        # However, Albumentations expects uint8 for some ops, float for Normalize.
        # We keep it uint8 here.

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic transform if none provided: Normalize + ToTensor
            # Note: This branch might not be used if get_dataloaders always provides transform
            basic_transform = A.Compose(
                [
                    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
            augmented = basic_transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.float32)


class DynamicBatchCollate:
    """
    Custom collate function that implements Dynamic Temporal Jittering.
    Resizes the entire batch to a random width sampled from config.JITTER_RANGE during training.
    """

    def __init__(self, mode="train", fixed_width=None):
        self.mode = mode
        self.fixed_width = fixed_width
        self.rng = np.random.default_rng(config.SEED)

    def __call__(self, batch):
        images, labels = zip(*batch)

        # Stack images: (B, C, H, W_orig)
        # Assuming all images from __getitem__ have the same size (256, 1246)
        images = torch.stack(images)
        labels = torch.stack(labels)

        # Determine target width
        if self.fixed_width is not None:
            target_width = self.fixed_width
        elif self.mode == "train":
            target_width = self.rng.integers(
                config.JITTER_RANGE[0], config.JITTER_RANGE[1] + 1
            )
        else:
            target_width = config.IMG_WIDTH_TEST

        # Resize
        # F.interpolate expects (N, C, H, W)
        # We use bilinear interpolation for spectrograms
        if images.shape[3] != target_width:
            images = F.interpolate(
                images,
                size=(config.IMG_HEIGHT, target_width),
                mode="bilinear",
                align_corners=False,
            )

        return images, labels


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


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),  # Time inversion
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders(use_pseudo_labels=False, load_cached_data=True, tta_width=None):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        use_pseudo_labels (bool): Whether to include pseudo-labeled test data in training.
        load_cached_data (bool): Whether to use cached metadata.
        tta_width (int, optional): If provided, overrides the test loader width for TTA.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = load_metadata(
        "train", use_pseudo_labels=use_pseudo_labels, load_cached_data=load_cached_data
    )
    val_df = load_metadata("val", load_cached_data=load_cached_data)
    test_df = load_metadata("test", load_cached_data=load_cached_data)

    # Datasets
    train_dataset = BirdDataset(
        train_df, mode="train", transform=get_transforms("train")
    )
    val_dataset = BirdDataset(val_df, mode="val", transform=get_transforms("val"))
    test_dataset = BirdDataset(test_df, mode="test", transform=get_transforms("test"))

    # Collate Functions
    train_collate = DynamicBatchCollate(mode="train")
    val_collate = DynamicBatchCollate(mode="val")

    # Test Collate: Support TTA width if provided
    test_width = tta_width if tta_width is not None else config.IMG_WIDTH_TEST
    test_collate = DynamicBatchCollate(mode="test", fixed_width=test_width)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=train_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=val_collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=test_collate,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
