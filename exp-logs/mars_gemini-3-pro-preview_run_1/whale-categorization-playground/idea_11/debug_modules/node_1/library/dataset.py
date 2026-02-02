import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Whale Identification.
    Reads images from disk and applies transformations.
    """

    def __init__(self, df, transform=None, class_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Image, file_path, [Id]).
            transform (albumentations.Compose): Augmentation pipeline.
            class_to_idx (dict): Mapping from class name to integer index.
            is_test (bool): If True, returns (image, image_name). If False, returns (image, label_idx).
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test

        # Pre-check file existence to avoid runtime errors
        self.valid_indices = []
        missing_count = 0
        for idx, row in self.df.iterrows():
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            if os.path.exists(full_path):
                self.valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            # In a real scenario we might log this, but we keep it silent as per instructions
            pass

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the dataset index to the dataframe index
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]

        # Load Image
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for corrupt images - create black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            augmented = base_transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and filename for submission mapping
            return image, row["Image"]
        else:
            # Return image and label index
            label_name = row["Id"]
            label_idx = self.class_to_idx[label_name]
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline based on the phase.

    Args:
        phase (str): 'train' or 'val'/'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                # Geometric Augmentations (Conservative)
                A.ShiftScaleRotate(
                    shift_limit=0.0,  # No shift
                    scale_limit=(
                        Config.AUG_SCALE_MIN - 1.0,
                        Config.AUG_SCALE_MAX - 1.0,
                    ),
                    rotate_limit=Config.AUG_ROTATION,
                    p=0.8,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=Config.AUG_HFLIP_PROB),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS,
                    contrast_limit=Config.AUG_CONTRAST,
                    p=0.5,
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_label_encoder(df_train, load_cached_data=True):
    """
    Generates or loads the class-to-index mapping.
    Strictly follows the caching logic requirement.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "classes.npy")

    classes = None

    # 1. Try to load cached data
    if load_cached_data:
        if os.path.exists(cache_file):
            try:
                classes = np.load(cache_file, allow_pickle=True)
            except Exception:
                pass  # Fall through to recompute

    # 2. Compute if not loaded
    if classes is None:
        # Extract unique IDs from training dataframe
        unique_ids = sorted(df_train["Id"].unique().tolist())
        classes = np.array(unique_ids)

        # Save to cache
        np.save(cache_file, classes)

    # Create mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    return class_to_idx, idx_to_class


def get_dataloaders(load_cached_data=True, verbose=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached label encoding.
        verbose (bool): Whether to print dataset stats.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx, idx_to_class
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Debugging: Reduce dataset size if configured
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)

        # Cite debug_lesson_1: Ensure validation set classes are present in training set
        valid_ids = set(df_train["Id"].unique())
        df_val = df_val[df_val["Id"].isin(valid_ids)]

        # Fallback if no overlap to prevent empty validation set
        if len(df_val) == 0:
            df_val = df_train.copy()

        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Get Label Encoder (Cached)
    # We pass df_train to generate classes if cache doesn't exist.
    # Note: df_train contains all classes (including singletons) as per split logic.
    class_to_idx, idx_to_class = get_label_encoder(
        df_train, load_cached_data=load_cached_data
    )

    # Define Transforms
    train_transform = get_transforms(phase="train")
    val_transform = get_transforms(phase="val")  # Same as test

    # Create Datasets
    train_dataset = WhaleDataset(
        df_train, transform=train_transform, class_to_idx=class_to_idx, is_test=False
    )

    val_dataset = WhaleDataset(
        df_val, transform=val_transform, class_to_idx=class_to_idx, is_test=False
    )

    test_dataset = WhaleDataset(
        df_test, transform=val_transform, class_to_idx=None, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    if verbose:
        print(f"DataLoaders created.")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches.")
        print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches.")
        print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches.")
        print(f"  Classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
