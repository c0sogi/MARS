import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


def get_transforms(stage: str = "train"):
    """
    Returns albumentations transforms for the specified stage.

    Args:
        stage (str): 'train', 'val', or 'test'.
    """
    if stage == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Moderate augmentation: RandomResizedCrop to capture details
                A.RandomResizedCrop(
                    height=Config.IMAGE_SIZE,
                    width=Config.IMAGE_SIZE,
                    scale=(0.75, 1.0),
                    ratio=(0.75, 1.333),
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    """
    Dataset class for Artwork Attribute Labeling.
    Handles loading images, processing multi-label targets, and optional soft labels.
    """

    def __init__(self, df, input_dir, transforms=None, soft_labels=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            input_dir (str): Root directory for images (usually ./input).
            transforms (albumentations.Compose): Transforms to apply.
            soft_labels (np.ndarray, optional): Soft targets for distillation (N, num_classes).
            is_test (bool): Whether this is the test set (no ground truth labels).
        """
        self.df = df
        self.input_dir = input_dir
        self.transforms = transforms
        self.soft_labels = soft_labels
        self.is_test = is_test

        # Pre-process file paths to avoid overhead in __getitem__
        # The metadata file_path is relative to input_dir
        self.file_paths = df["file_path"].values
        self.ids = df["id"].values

        if not self.is_test:
            # Pre-process labels
            self.labels = []
            for attr_str in df["attribute_ids"].values:
                if pd.isna(attr_str) or attr_str == "":
                    self.labels.append([])
                else:
                    self.labels.append([int(x) for x in str(attr_str).split()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Read with OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should not happen given metadata validation)
            # Create a black image to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        result = {"image": image, "id": self.ids[idx]}

        if not self.is_test:
            # Create Multi-Hot Target
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            label_indices = self.labels[idx]
            if len(label_indices) > 0:
                target[label_indices] = 1.0
            result["target"] = target

            # Add Soft Labels if available
            if self.soft_labels is not None:
                soft_target = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
                result["soft_target"] = soft_target

        return result


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    soft_labels_path=None,
    debug=Config.DEBUG,
    batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.VAL_BATCH_SIZE,
):
    """
    Factory function to create DataLoaders.

    Args:
        train_metadata_path (str): Path to train metadata CSV.
        val_metadata_path (str): Path to val metadata CSV.
        test_metadata_path (str): Path to test metadata CSV.
        soft_labels_path (str, optional): Path to .npy file containing soft labels for training data.
        debug (bool): If True, subsets data for quick debugging.
        batch_size (int): Training batch size.
        val_batch_size (int): Validation/Test batch size.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    dataloaders = {}

    # --- Train Loader ---
    if os.path.exists(train_metadata_path):
        df_train = pd.read_csv(train_metadata_path)

        # Load Soft Labels if provided
        soft_labels = None
        if soft_labels_path and os.path.exists(soft_labels_path):
            logger.info(f"Loading soft labels from {soft_labels_path}")
            try:
                soft_labels = np.load(soft_labels_path)
                if len(soft_labels) != len(df_train):
                    logger.warning(
                        f"Soft labels shape {soft_labels.shape} does not match train metadata len {len(df_train)}. "
                        "Ignoring soft labels."
                    )
                    soft_labels = None
            except Exception as e:
                logger.error(f"Failed to load soft labels: {e}")
                soft_labels = None

        if debug:
            logger.info(
                f"Debug mode: Subsetting train data to {Config.DEBUG_SUBSET_SIZE}"
            )
            df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]
            if soft_labels is not None:
                soft_labels = soft_labels[: Config.DEBUG_SUBSET_SIZE]

        train_dataset = ArtworkDataset(
            df=df_train,
            input_dir=Config.INPUT_DIR,
            transforms=get_transforms("train"),
            soft_labels=soft_labels,
            is_test=False,
        )

        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        logger.info(f"Train DataLoader created: {len(df_train)} samples")
    else:
        logger.warning(f"Train metadata not found at {train_metadata_path}")

    # --- Val Loader ---
    if os.path.exists(val_metadata_path):
        df_val = pd.read_csv(val_metadata_path)
        if debug:
            df_val = df_val.iloc[: Config.DEBUG_SUBSET_SIZE]

        val_dataset = ArtworkDataset(
            df=df_val,
            input_dir=Config.INPUT_DIR,
            transforms=get_transforms("val"),
            is_test=False,
        )

        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        logger.info(f"Val DataLoader created: {len(df_val)} samples")
    else:
        logger.warning(f"Val metadata not found at {val_metadata_path}")

    # --- Test Loader ---
    if os.path.exists(test_metadata_path):
        df_test = pd.read_csv(test_metadata_path)
        # No debug subsetting for test usually, unless explicitly requested for pipeline check
        if debug:
            df_test = df_test.iloc[:100]

        test_dataset = ArtworkDataset(
            df=df_test,
            input_dir=Config.INPUT_DIR,
            transforms=get_transforms("test"),
            is_test=True,
        )

        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        logger.info(f"Test DataLoader created: {len(df_test)} samples")
    else:
        logger.warning(f"Test metadata not found at {test_metadata_path}")

    return dataloaders
