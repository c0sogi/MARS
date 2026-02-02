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


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'val' (also used for test).

    Returns:
        A.Compose: The composed transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                # Aspect Ratio Preserving Resize
                A.LongestMaxSize(max_size=Config.IMG_SIZE),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Geometric Augmentations Only (No Cutout/Occlusion)
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Gallery transforms
        return A.Compose(
            [
                # Aspect Ratio Preserving Resize
                A.LongestMaxSize(max_size=Config.IMG_SIZE),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    """
    Custom Dataset for Whale Identification.
    """

    def __init__(self, df, transforms=None, label_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transforms to apply.
            label_encoder (LabelEncoder): Fitted LabelEncoder for mapping IDs.
            is_test (bool): If True, returns image ID instead of label.
        """
        self.df = df
        self.transforms = transforms
        self.label_encoder = label_encoder
        self.is_test = is_test

        # Pre-calculate full file paths
        # Metadata 'file_path' is relative (e.g., "train/img.jpg")
        # We join with INPUT_ROOT (e.g., "./input")
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, fp) for fp in df["file_path"].values
        ]

        # Store IDs if available and not testing
        if not self.is_test and "Id" in df.columns:
            self.ids = df["Id"].values
        else:
            self.ids = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load Image
        image = cv2.imread(path)
        if image is None:
            # Fallback for robustness (should not trigger given validation)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return Logic
        if self.is_test:
            # For test set, return image and the original filename (Image ID)
            return image, self.df.iloc[idx]["Image"]

        target = self.ids[idx]

        if self.label_encoder:
            # Encode label
            try:
                label = self.label_encoder.transform([target])[0]
            except ValueError:
                # Handle unseen labels (should be filtered out upstream)
                label = -1

            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Return raw label
            return image, target


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to load cached LabelEncoder classes.
        debug (bool): If True, subsets data for quick iteration.

    Returns:
        tuple: (train_loader, val_loader, label_encoder)
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # 2. Filter out 'new_whale'
    # We only train on known identities.
    df_train = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)
    df_val = df_val[df_val["Id"] != "new_whale"].reset_index(drop=True)

    # 3. Debug Subsetting
    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 4. Label Encoding with Caching
    le_path = os.path.join(Config.WORKING_DIR, "label_encoder_classes.npy")
    le = LabelEncoder()

    if load_cached_data and os.path.exists(le_path):
        classes = np.load(le_path, allow_pickle=True)
        le.classes_ = classes
    else:
        # Fit on all known training IDs
        le.fit(df_train["Id"].unique())
        # Cache the classes
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(le_path, le.classes_)

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        df_train, transforms=get_transforms("train"), label_encoder=le
    )

    val_dataset = WhaleDataset(
        df_val, transforms=get_transforms("val"), label_encoder=le
    )

    # 6. Create DataLoaders
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

    return train_loader, val_loader, le


def get_inference_gallery_loader(load_cached_data=True, debug=False):
    """
    Creates a DataLoader for the Training set to be used as a Gallery
    during inference/validation (No Augmentation, No Shuffle).

    Args:
        load_cached_data (bool): Whether to load cached LabelEncoder classes.
        debug (bool): Debug mode.

    Returns:
        DataLoader: Loader for the gallery.
    """
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_train = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Load Label Encoder
    le_path = os.path.join(Config.WORKING_DIR, "label_encoder_classes.npy")
    le = LabelEncoder()
    if load_cached_data and os.path.exists(le_path):
        classes = np.load(le_path, allow_pickle=True)
        le.classes_ = classes
    else:
        le.fit(df_train["Id"].unique())

    dataset = WhaleDataset(
        df_train, transforms=get_transforms("val"), label_encoder=le  # Clean transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader


def get_test_loader(debug=False):
    """
    Creates a DataLoader for the Test set.

    Args:
        debug (bool): Debug mode.

    Returns:
        DataLoader: Loader for the test set.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    dataset = WhaleDataset(
        df_test, transforms=get_transforms("val"), is_test=True  # Clean transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
