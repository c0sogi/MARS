import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles dynamic loading of spectrograms, resizing, channel replication,
    and hybrid label handling (ground truth + pseudo-labels).
    """

    def __init__(
        self, df, img_dir, num_classes=19, transform=None, pseudo_labels_dict=None
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            img_dir (str): Directory containing spectrogram BMPs.
            num_classes (int): Number of classes (species) to load.
            transform (albumentations.Compose): Transforms to apply.
            pseudo_labels_dict (dict, optional): Dictionary mapping rec_id to soft label vectors.
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.pseudo_labels_dict = pseudo_labels_dict or {}

        # Identify label columns explicitly to avoid artifacts Cite {debug_lesson_1}
        self.label_cols = [f"species_{i}" for i in range(num_classes)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Resolve image path
        # Metadata file_path is like "essential_data/src_wavs/PC10_... .wav"
        # Spectrograms are in img_dir with same basename but .bmp extension
        wav_path = row["file_path"]
        basename = os.path.basename(wav_path)
        bmp_name = os.path.splitext(basename)[0] + ".bmp"
        img_path = os.path.join(self.img_dir, bmp_name)

        # Load Image
        # Load as grayscale (1 channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (should not happen in valid dataset)
            # Create a blank image of expected size to avoid crashing
            image = np.zeros((256, 640), dtype=np.uint8)

        # Resize to High-Fidelity Resolution (Width=640, Height=256)
        # cv2.resize takes (width, height)
        image = cv2.resize(image, (640, 256), interpolation=cv2.INTER_LINEAR)

        # Channel Replication: 1ch -> 3ch for ResNet
        image = cv2.merge([image, image, image])

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Handle Labels
        if rec_id in self.pseudo_labels_dict:
            # Use soft pseudo-labels
            target = self.pseudo_labels_dict[rec_id]
            target = torch.tensor(target, dtype=torch.float32)
        else:
            # Use ground truth from dataframe
            target = row[self.label_cols].values.astype(np.float32)
            target = torch.tensor(target, dtype=torch.float32)

        return image, target


def get_transforms(cfg, mode="train"):
    """
    Returns the Albumentations transforms for training or validation/testing.

    Args:
        cfg (Config): Configuration object.
        mode (str): "train" or "val".

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        transforms = [
            # Robust Augmentation Strategy
            # 1. Horizontal Flip (Time Inversion)
            A.HorizontalFlip(p=0.5 if cfg.USE_HORIZONTAL_FLIP else 0.0),
            # 2. Unstructured Cutout (CoarseDropout)
            # Randomly mask out rectangular regions
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5 if cfg.USE_CUTOUT else 0.0,
            ),
            # Normalization and Tensor Conversion
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


class Mixup:
    """
    Mixup implementation for batch processing.
    Mixes images and targets with a Beta distribution.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.rng = np.random.default_rng()

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x (torch.Tensor): Input images (B, C, H, W).
            batch_y (torch.Tensor): Targets (B, NumClasses).

        Returns:
            mixed_x, mixed_y
        """
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        # Mix images
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]

        # Mix targets (works for multi-label soft/binary targets)
        mixed_y = lam * batch_y + (1 - lam) * batch_y[index, :]

        return mixed_x, mixed_y


def get_dataloaders(cfg, stage="teacher", pseudo_labels=None):
    """
    Constructs DataLoaders for different stages of the pipeline.

    Args:
        cfg (Config): Configuration object.
        stage (str): 'teacher', 'inference', or 'student'.
        pseudo_labels (np.ndarray, optional): Soft labels for the test set (used in 'student' stage).

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' dataloaders as appropriate.
    """
    set_seed(cfg.SEED)

    loaders = {}

    # Load Metadata
    df_train = pd.read_csv(cfg.TRAIN_CSV)
    df_val = pd.read_csv(cfg.VAL_CSV)
    df_test = pd.read_csv(cfg.TEST_CSV)

    # Debugging: Subset data
    if cfg.MAX_SAMPLES:
        df_train = df_train.head(cfg.MAX_SAMPLES)
        df_val = df_val.head(cfg.MAX_SAMPLES)
        df_test = df_test.head(cfg.MAX_SAMPLES)
        if pseudo_labels is not None:
            pseudo_labels = pseudo_labels[: cfg.MAX_SAMPLES]

    # Common Transforms
    train_transform = get_transforms(cfg, mode="train")
    val_transform = get_transforms(cfg, mode="val")

    if stage == "teacher":
        # Stage 1: Train on Labeled Train, Validate on Labeled Val
        train_dataset = BirdDataset(
            df_train,
            cfg.SPECTROGRAM_DIR,
            num_classes=cfg.NUM_CLASSES,
            transform=train_transform,
        )
        val_dataset = BirdDataset(
            df_val,
            cfg.SPECTROGRAM_DIR,
            num_classes=cfg.NUM_CLASSES,
            transform=val_transform,
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

    elif stage == "inference":
        # Stage 2: Inference on Test Set (for pseudo-labeling or submission)
        test_dataset = BirdDataset(
            df_test,
            cfg.SPECTROGRAM_DIR,
            num_classes=cfg.NUM_CLASSES,
            transform=val_transform,
        )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

    elif stage == "student":
        # Stage 3: Train on Combined (Train + Test w/ Pseudo), Validate on Labeled Val
        if pseudo_labels is None:
            raise ValueError("pseudo_labels must be provided for student stage.")

        # Create Pseudo Label Dictionary
        # Assumes pseudo_labels array aligns with df_test rows
        pseudo_dict = {}
        for idx, row in df_test.iterrows():
            rec_id = row["rec_id"]
            # If we subsetted for debug, pseudo_labels is also subsetted
            if idx < len(pseudo_labels):
                pseudo_dict[rec_id] = pseudo_labels[idx]

        # Combine DataFrames
        df_combined = pd.concat([df_train, df_test], ignore_index=True)

        train_dataset = BirdDataset(
            df_combined,
            cfg.SPECTROGRAM_DIR,
            num_classes=cfg.NUM_CLASSES,
            transform=train_transform,
            pseudo_labels_dict=pseudo_dict,
        )

        val_dataset = BirdDataset(
            df_val,
            cfg.SPECTROGRAM_DIR,
            num_classes=cfg.NUM_CLASSES,
            transform=val_transform,
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

    return loaders
