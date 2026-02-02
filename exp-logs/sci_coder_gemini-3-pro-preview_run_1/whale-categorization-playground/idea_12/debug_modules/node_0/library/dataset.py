import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(phase="train"):
    """
    Returns the albumentations transform pipeline based on the phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Conservative Affine: Rotation +/- 20, Scale 0.9-1.1 (scale_limit=0.1)
                # shift_limit is kept at 0 to avoid cutting off the fluke
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # Photometric: Brightness and Contrast only
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_classes(load_cached_data=True):
    """
    Retrieves the list of unique classes (whale Ids).
    Implements caching mechanism using .npy format.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.array: Array of unique class names.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "classes.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True)
            return classes
        except Exception as e:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    # We derive classes strictly from the training metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Get unique Ids and sort them for determinism
    classes = np.unique(df_train["Id"].values)
    classes.sort()

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, classes)

    return classes


class WhaleDataset(Dataset):
    def __init__(self, df, transforms=None, class_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transforms to apply.
            class_to_idx (dict): Mapping from class name to integer index.
            is_test (bool): If True, returns image and image_name. If False, returns image and label.
        """
        self.df = df
        self.transforms = transforms
        self.class_to_idx = class_to_idx
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # metadata file_path is relative to input dir (e.g., "train/img.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should be caught by metadata checks, but for safety)
            # Create a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            # For test, return image and the filename (Image ID)
            return image, row["Image"]
        else:
            # For train/val, return image and label index
            label_name = row["Id"]
            label_idx = self.class_to_idx[label_name]
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Creates DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached class definitions.
        debug_size (int, optional): If provided, subsets the data for debugging.

    Returns:
        tuple: (train_loader, val_loader, class_list)
    """
    # 1. Get Classes
    classes = get_classes(load_cached_data=load_cached_data)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # 2. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Debug mode
    if debug_size is not None:
        df_train = df_train.iloc[:debug_size]
        df_val = df_val.iloc[:debug_size]

    # 3. Create Datasets
    train_dataset = WhaleDataset(
        df=df_train,
        transforms=get_transforms(phase="train"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df=df_val,
        transforms=get_transforms(phase="val"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to ensure batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for testing/inference.

    Args:
        load_cached_data (bool): Used to retrieve consistent class list if needed for post-processing.

    Returns:
        tuple: (test_loader, class_list)
    """
    # We need the class list to map predictions back to strings later
    classes = get_classes(load_cached_data=load_cached_data)

    df_test = pd.read_csv(Config.TEST_CSV)

    test_dataset = WhaleDataset(
        df=df_test,
        transforms=get_transforms(phase="test"),
        class_to_idx=None,  # Not needed for test
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, classes
