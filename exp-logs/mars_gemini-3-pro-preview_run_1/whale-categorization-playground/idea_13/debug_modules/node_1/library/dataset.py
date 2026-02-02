import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transformation pipeline for a specific data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    size = Config.IMAGE_SIZE

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                # Conservative Affine Transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                # Flip
                A.HorizontalFlip(p=0.5),
                # Photometric Distortions (No Hue/Saturation)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization
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
                A.Resize(height=size, width=size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Whale Identification.
    """

    def __init__(self, df, transform=None, label_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'Image', 'file_path', and 'Id' (if not test).
            transform (A.Compose): Albumentations transforms.
            label_encoder (LabelEncoder): Fitted sklearn LabelEncoder.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test

        # Pre-fetch columns to numpy arrays for speed
        self.file_paths = df["file_path"].values
        self.image_names = df["Image"].values

        if not self.is_test:
            self.ids = df["Id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load Image
        image = cv2.imread(full_path)

        # Safety check for missing/corrupt images
        if image is None:
            # Return a black image of correct size to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.is_test:
            # Return image and image_name (for submission file)
            return image, self.image_names[idx]

        # Training/Validation mode
        label_str = self.ids[idx]
        # Encode label
        # Note: We assume label_encoder covers all classes in training/val
        label_idx = self.label_encoder.transform([label_str])[0]

        return image, torch.tensor(label_idx, dtype=torch.long), self.image_names[idx]


def get_label_encoder(load_cached_data=True):
    """
    Manages the LabelEncoder with caching to ensure consistency and speed.

    Args:
        load_cached_data (bool): If True, attempts to load classes from disk.

    Returns:
        LabelEncoder: Fitted sklearn LabelEncoder.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "classes.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True)
            le = LabelEncoder()
            le.classes_ = classes
            return le
        except Exception as e:
            print(f"Failed to load cached classes: {e}. Recomputing...")

    # 2. Compute from scratch
    # We strictly use the original training set to define the class mapping
    df_train = pd.read_csv(Config.TRAIN_CSV)
    unique_ids = sorted(df_train["Id"].unique())

    le = LabelEncoder()
    le.fit(unique_ids)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, le.classes_)

    return le


def get_loaders(load_cached_data=True, extra_train_df=None):
    """
    Factory function to create DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached label encoder.
        extra_train_df (pd.DataFrame, optional): Additional labeled data (e.g., pseudo-labels)
                                                 to merge with the training set.

    Returns:
        tuple: (train_loader, val_loader, test_loader, label_encoder)
    """
    # 1. Prepare Label Encoder
    le = get_label_encoder(load_cached_data)

    # 2. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 3. Handle Debug Mode
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 4. Handle Self-Training (Stage 2)
    if extra_train_df is not None:
        # Concatenate original training data with new pseudo-labeled data
        # We reset index to ensure __getitem__ works correctly
        df_train = pd.concat([df_train, extra_train_df], axis=0).reset_index(drop=True)

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        df_train, transform=get_transforms("train"), label_encoder=le, is_test=False
    )

    val_dataset = WhaleDataset(
        df_val, transform=get_transforms("val"), label_encoder=le, is_test=False
    )

    test_dataset = WhaleDataset(
        df_test, transform=get_transforms("test"), label_encoder=None, is_test=True
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    # Test loader often uses larger batch size as no gradients are needed
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.get_batch_size(inference=True),
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, le
