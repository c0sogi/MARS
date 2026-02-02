import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import library.config as config
import library.utils as utils


class WhaleDataset(Dataset):
    """
    Custom Dataset for loading Whale images.
    """

    def __init__(self, df, root_dir, transform=None, label_map=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Image, Id, file_path).
            root_dir (str): Root directory for images (usually config.INPUT_DIR).
            transform (albumentations.Compose): Transformations to apply.
            label_map (dict): Dictionary mapping string Ids to integer labels.
            is_test (bool): If True, returns (image, filename). If False, returns (image, label, label_str).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        self.is_test = is_test

        # Pre-compute full file paths
        # metadata file_path is relative (e.g., "train/img.jpg")
        self.file_paths = [os.path.join(root_dir, fp) for fp in df["file_path"].values]

        if not self.is_test:
            self.ids = df["Id"].values
            self.filenames = df["Image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(path)

        # Handle potential missing or corrupt images gracefully
        if image is None:
            # Return a black image of the expected size
            image = np.zeros(
                (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to basic tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            # For test set, we need the filename to create the submission
            filename = self.df.iloc[idx]["Image"]
            return image, filename
        else:
            label_str = self.ids[idx]
            # Map string label to integer
            # If label_map is provided, use it. Otherwise default to -1.
            label = self.label_map.get(label_str, -1) if self.label_map else -1
            return image, label, label_str


def get_transforms(img_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        img_size (tuple): (height, width)
        mode (str): "train" or "eval"
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                # Geometric augmentations to make the model robust to pose variations
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Add CoarseDropout (Cutout) for regularization
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
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Eval / Test / Gallery
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Prepares and returns DataLoaders for Train, Val, Gallery, and Test sets.

    Returns:
        train_loader, val_loader, gallery_loader, test_loader, label_map, num_classes
    """
    # 1. Load Metadata
    df_train_raw = pd.read_csv(config.TRAIN_CSV)
    df_val_raw = pd.read_csv(config.VAL_CSV)
    df_test = pd.read_csv(config.TEST_CSV)

    # 2. Filter Data for ArcFace Training
    # ArcFace is trained to discriminate between KNOWN identities.
    # 'new_whale' is not a specific identity, so we exclude it from the classification training.
    # We create filtered DataFrames for Train and Val.
    df_train_known = (
        df_train_raw[df_train_raw["Id"] != "new_whale"].copy().reset_index(drop=True)
    )
    df_val_known = (
        df_val_raw[df_val_raw["Id"] != "new_whale"].copy().reset_index(drop=True)
    )

    # 3. Create Label Mapping
    # We collect all unique IDs from the known training set.
    unique_ids = sorted(df_train_known["Id"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_ids)}
    num_classes = len(unique_ids)

    print(f"Data Loading Summary:")
    print(f"  Known Classes: {num_classes}")
    print(f"  Train Samples (Known): {len(df_train_known)}")
    print(f"  Val Samples (Known): {len(df_val_known)}")
    print(f"  Test Samples: {len(df_test)}")

    # 4. Define Transforms
    train_transform = get_transforms(config.IMAGE_SIZE, mode="train")
    eval_transform = get_transforms(config.IMAGE_SIZE, mode="eval")

    # 5. Create Datasets

    # Train Dataset: Used for training the ArcFace model
    train_dataset = WhaleDataset(
        df_train_known,
        config.INPUT_DIR,
        transform=train_transform,
        label_map=label_map,
        is_test=False,
    )

    # Validation Dataset: Used for monitoring classification accuracy on knowns
    val_dataset = WhaleDataset(
        df_val_known,
        config.INPUT_DIR,
        transform=eval_transform,
        label_map=label_map,
        is_test=False,
    )

    # Gallery Dataset: Used for building the reference embedding database.
    # It contains ALL known whale images (from the training set).
    # We use eval_transform (no augmentation) to get deterministic embeddings.
    gallery_dataset = WhaleDataset(
        df_train_known,
        config.INPUT_DIR,
        transform=eval_transform,
        label_map=label_map,
        is_test=False,
    )

    # Test Dataset: Used for final inference
    test_dataset = WhaleDataset(
        df_test,
        config.INPUT_DIR,
        transform=eval_transform,
        label_map=None,
        is_test=True,
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Recommended for BatchNorm stability and ArcFace margin consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
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

    return train_loader, val_loader, gallery_loader, test_loader, label_map, num_classes
