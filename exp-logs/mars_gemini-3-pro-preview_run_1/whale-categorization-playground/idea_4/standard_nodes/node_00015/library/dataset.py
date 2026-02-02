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


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads the mapping between class names (Ids) and integer labels.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        np.ndarray: Array of class names where index corresponds to label.
        dict: Mapping from class name to integer label.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path, allow_pickle=True)
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            return classes, class_to_idx
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Get unique IDs and sort them for determinism
    # 'new_whale' is just another class in the set
    unique_ids = sorted(df_train["Id"].unique())
    classes = np.array(unique_ids)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, classes)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    return classes, class_to_idx


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                # Conservative Affine: Rotation +/- 20, Scale 0.9-1.1
                # ShiftScaleRotate is a convenient wrapper. shift_limit=0 disables translation.
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentations (Excluding Hue/Saturation)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, class_to_idx=None, transform=None, mode="train"
    ):
        """
        Args:
            csv_file (str): Path to the metadata CSV.
            root_dir (str): Root directory of the input data.
            class_to_idx (dict, optional): Mapping from Id string to int. Required for train/val.
            transform (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Debug mode: subset data
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SUBSET_SIZE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata csv contains 'file_path' relative to input dir
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (should be rare given metadata checks)
            # Create a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            # For test, we need the filename to identify predictions
            return image, row["Image"]
        else:
            # For train/val, we need the label index
            label_str = row["Id"]
            label = self.class_to_idx[label_str]
            return image, label


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached class mapping.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    # 1. Prepare Class Mapping
    classes, class_to_idx = get_class_mapping(load_cached_data=load_cached_data)

    # 2. Define Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # 3. Instantiate Datasets
    train_dataset = WhaleDataset(
        csv_file=Config.TRAIN_CSV,
        root_dir=Config.INPUT_DIR,
        class_to_idx=class_to_idx,
        transform=train_transform,
        mode="train",
    )

    val_dataset = WhaleDataset(
        csv_file=Config.VAL_CSV,
        root_dir=Config.INPUT_DIR,
        class_to_idx=class_to_idx,
        transform=val_transform,
        mode="val",
    )

    test_dataset = WhaleDataset(
        csv_file=Config.TEST_CSV,
        root_dir=Config.INPUT_DIR,
        class_to_idx=None,  # Not needed for test
        transform=test_transform,
        mode="test",
    )

    # 4. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, classes
