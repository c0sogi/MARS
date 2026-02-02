import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import load_data_splits


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Loads BMP spectrograms, processes them to 3-channel RGB, and handles multi-label targets.
    """

    def __init__(self, df, config, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transforms = transforms
        self.mode = mode

        # Pre-parse labels for efficiency if in train/val mode
        if self.mode != "test":
            self.labels = self._process_labels()

    def _process_labels(self):
        label_matrix = np.zeros(
            (len(self.df), self.config.NUM_CLASSES), dtype=np.float32
        )
        for idx, row in self.df.iterrows():
            label_str = str(row["labels"])
            if pd.isna(label_str) or label_str == "?" or label_str.strip() == "":
                continue

            try:
                indices = [int(x) for x in label_str.split()]
                for class_idx in indices:
                    if 0 <= class_idx < self.config.NUM_CLASSES:
                        label_matrix[idx, class_idx] = 1.0
            except ValueError:
                continue
        return label_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert wav path to bmp path
        # Metadata file_path: essential_data/src_wavs/filename.wav
        # Spectrogram dir: supplemental_data/spectrograms/filename.bmp
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = os.path.splitext(wav_filename)[0] + ".bmp"
        image_path = os.path.join(self.config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing files (should not happen based on metadata check)
            image = np.zeros(self.config.IMAGE_SIZE, dtype=np.uint8)

        # Resize
        image = cv2.resize(image, self.config.IMAGE_SIZE)

        # Replicate to 3 channels (Pseudo-RGB)
        image = cv2.merge([image, image, image])

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transforms provided (should minimally be ToTensor)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return logic
        if self.mode == "test":
            return image, row["rec_id"]
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label


def get_transforms(data, config):
    """
    Returns the Albumentations transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=config.BRIGHTNESS_LIMIT,
                    contrast_limit=config.CONTRAST_LIMIT,
                    p=0.5,
                ),
                # Horizontal Translation (Time-shifting)
                # border_mode=cv2.BORDER_CONSTANT with value=0 ensures zero-padding
                A.ShiftScaleRotate(
                    shift_limit=config.SHIFT_LIMIT,
                    scale_limit=0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Normalization (ImageNet stats)
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


class MixupCollator:
    """
    Collator that applies Mixup regularization to a batch of data.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)

        batch_size = images.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]

        return mixed_images, mixed_labels


def make_folds(config, load_cached_data=True):
    """
    Wrapper to load data splits using the configuration library.
    """
    return load_data_splits(config, load_cached_data=load_cached_data)


def get_dataloaders(fold_idx, folds_df, test_df, config):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): The current fold index (0-4).
        folds_df (pd.DataFrame): The dataframe containing training data and 'kfold' info.
        test_df (pd.DataFrame): The dataframe containing test data.
        config (Config): Configuration object.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Split Train/Val based on kfold
    train_df = folds_df[folds_df["kfold"] != fold_idx].reset_index(drop=True)
    val_df = folds_df[folds_df["kfold"] == fold_idx].reset_index(drop=True)

    # Datasets
    train_dataset = BirdDataset(
        train_df, config, transforms=get_transforms("train", config), mode="train"
    )
    val_dataset = BirdDataset(
        val_df, config, transforms=get_transforms("val", config), mode="val"
    )
    test_dataset = BirdDataset(
        test_df, config, transforms=get_transforms("test", config), mode="test"
    )

    # Collator for training
    train_collator = (
        MixupCollator(alpha=config.MIXUP_ALPHA) if config.USE_MIXUP else None
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=train_collator,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
