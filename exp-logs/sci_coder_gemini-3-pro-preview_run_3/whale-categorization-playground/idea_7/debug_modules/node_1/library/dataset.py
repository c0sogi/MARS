import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


# -------------------------------------------------------------------------
# Augmentation & Transforms
# -------------------------------------------------------------------------
def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Strategy:
    - Anisotropic Resizing: Force image to 512x512, ignoring aspect ratio.
    - Geometric Augmentations: Rotation, scaling, shear, flip (Train only).
    - No Occlusion: Avoid Cutout/RandomErasing to preserve singleton features.
    """
    if mode == "train":
        return A.Compose(
            [
                # Anisotropic Resizing (Squashing)
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # Normalization & Tensor Conversion
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------
class WhaleDataset(Dataset):
    """
    Custom Dataset for Whale Identification.

    Handles:
    - Loading images from disk.
    - Converting BGR to RGB.
    - Applying transforms.
    - Returning (image, label_idx) for Train/Val.
    - Returning (image, filename) for Test.
    """

    def __init__(self, df, mode="train", transforms=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

        # Pre-calculate full paths
        # df['file_path'] is relative (e.g., "train/img.jpg")
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in self.df["file_path"]
        ]

        # Prepare targets
        if self.mode in ["train", "val"]:
            self.targets = self.df["label_idx"].values
        else:
            # For test, we track filenames to map predictions back to Image ID
            self.targets = self.df["Image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load Image
        img = cv2.imread(path)
        if img is None:
            # Fallback for missing/corrupt images (should not happen based on metadata check)
            # Create a black image of target size
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        target = self.targets[idx]

        # Return format:
        # Train/Val: (Tensor, int_label)
        # Test: (Tensor, str_filename)
        return img_tensor, target


# -------------------------------------------------------------------------
# Data Loading & Caching Logic
# -------------------------------------------------------------------------
def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Implements caching for the Label Encoder to ensure deterministic class mapping.

    Args:
        debug (bool): If True, subsamples data for quick debugging.
        load_cached_data (bool): If True, attempts to load processed metadata/encoder from cache.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Filter Training Data
    # Strategy: Exclude 'new_whale' from training to train on known identities only.
    df_train = df_train[df_train["Id"] != "new_whale"].copy()
    # Filter validation data to ensure all labels are known (exclude 'new_whale')
    df_val = df_val[df_val["Id"] != "new_whale"].copy()

    # 3. Label Encoding with Caching
    cache_path = os.path.join(Config.WORKING_DIR, "label_encoder_classes.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached label encoder classes from {cache_path}")
        unique_classes = np.load(cache_path)
    else:
        print("Computing label encoder classes from scratch...")
        unique_classes = np.sort(df_train["Id"].unique())
        np.save(cache_path, unique_classes)
        print(f"Saved label encoder classes to {cache_path}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    # 4. Map Labels
    # Map Train
    df_train["label_idx"] = df_train["Id"].map(class_to_idx)
    # Map Val (Note: Val set is guaranteed to be a subset of known classes by metadata logic)
    df_val["label_idx"] = df_val["Id"].map(class_to_idx)

    # Debugging Subsample
    if debug:
        print(f"Debug Mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        df_train, mode="train", transforms=get_transforms(mode="train")
    )

    val_dataset = WhaleDataset(
        df_val, mode="val", transforms=get_transforms(mode="val")
    )

    test_dataset = WhaleDataset(
        df_test, mode="test", transforms=get_transforms(mode="test")
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for stability with ArcFace
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(
        f"Data Loaded: Train({len(df_train)}), Val({len(df_val)}), Test({len(df_test)})"
    )
    print(f"Number of classes: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes
