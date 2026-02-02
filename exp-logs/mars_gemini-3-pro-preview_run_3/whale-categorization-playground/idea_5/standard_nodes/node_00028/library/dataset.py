import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                # Geometric augmentations: Rotation and Scaling
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
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


class WhaleDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, label_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory containing images (usually Config.INPUT_DIR).
            transform (A.Compose): Albumentations transforms.
            label_encoder (dict): Mapping from Id string to int. Required for train/val.
            is_test (bool): If True, returns dummy labels.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test

        # Pre-compute full paths to avoid overhead in __getitem__
        # Metadata file_path is relative (e.g., "train/img.jpg")
        self.file_paths = [os.path.join(root_dir, fp) for fp in df["file_path"].values]

        if not self.is_test:
            self.ids = df["Id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (though validation showed none missing)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Labels
        if self.is_test:
            # Return image and filename (for submission)
            filename = self.df.iloc[idx]["Image"]
            return image, filename
        else:
            label_str = self.ids[idx]
            label = self.label_encoder.get(label_str, -1)

            # Sanity check
            if label == -1:
                # This should not happen given the filtering logic in get_loaders
                # But if it does, return a dummy label 0
                label = 0

            return image, torch.tensor(label, dtype=torch.long)


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training, validation, and testing.
    Handles caching of the label encoder (classes).

    Args:
        load_cached_data (bool): If True, attempts to load processed classes from cache.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    classes_cache_path = os.path.join(Config.CACHE_DIR, "classes.npy")

    # 1. Load Metadata
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_val_raw = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Filter 'new_whale' for Training and Validation
    # Strategy: Train only on known identities.
    df_train = df_train_raw[df_train_raw["Id"] != "new_whale"].copy()
    df_val = df_val_raw[df_val_raw["Id"] != "new_whale"].copy()

    # 3. Handle Label Encoding (Caching Logic)
    classes = None
    if load_cached_data and os.path.exists(classes_cache_path):
        try:
            classes = np.load(classes_cache_path, allow_pickle=True)
            print(f"Loaded {len(classes)} classes from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating.")
            classes = None

    if classes is None:
        # Generate classes from Training data
        # We sort them to ensure deterministic mapping
        classes = np.unique(df_train["Id"].values)
        classes.sort()

        # Save to cache
        np.save(classes_cache_path, classes)
        print(f"Generated and cached {len(classes)} classes.")

    # Create mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    # 4. Debug Mode Subsampling
    if Config.DEBUG:
        print(f"Debug Mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE] if len(df_val) > 0 else df_val
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        df=df_train,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train"),
        label_encoder=class_to_idx,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df=df_val,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="val"),
        label_encoder=class_to_idx,
        is_test=False,
    )

    test_dataset = WhaleDataset(
        df=df_test,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="test"),
        label_encoder=None,
        is_test=True,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for batch norm stability with small batches
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

    print(f"Data Loaders created:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val:   {len(df_val)} samples")
    print(f"  Test:  {len(df_test)} samples")
    print(f"  Classes: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes
