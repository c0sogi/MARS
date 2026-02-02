import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_type: str = "train"):
    """
    Returns the Albumentations transformation pipeline based on the data type.

    Args:
        data_type (str): One of 'train', 'valid', 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Geometric Augmentations
                A.RandomRotate90(p=Config.AUG_ROTATE90_P),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_P,
                ),
                # Color & Regularization
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=Config.AUG_HSV_P,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=Config.AUG_BRIGHTNESS_CONTRAST_P,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMAGE_SIZE * 0.1),
                    max_width=int(Config.IMAGE_SIZE * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=Config.AUG_COARSE_DROPOUT_P,
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    elif data_type in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def load_data(
    csv_path: str, dataset_name: str, debug: bool = False, load_cached_data: bool = True
):
    """
    Loads the dataframe from CSV or Cache. Implements strict caching logic.

    Args:
        csv_path (str): Path to the source CSV file.
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        debug (bool): If True, subsamples the dataset.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_filename = f"{dataset_name}_processed.parquet"
    if debug:
        cache_filename = f"{dataset_name}_debug_processed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {dataset_name} data from cache: {cache_path}")
            return df
        except Exception as e:
            # print(f"Failed to load cache: {e}. Reloading from source.")
            pass

    # 2. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if debug:
        df = df.sample(n=min(100, len(df)), random_state=Config.SEED).reset_index(
            drop=True
        )

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {dataset_name} data to cache: {cache_path}")
    except Exception as e:
        # print(f"Warning: Could not save cache: {e}")
        pass

    return df


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading, augmentation, and multi-label target generation.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, output_label: bool = True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'file_path' and 'labels'.
            transforms (albumentations.Compose): Transformations to apply.
            output_label (bool): Whether to return target labels.
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label
        self.classes = Config.CLASSES
        self.num_classes = len(self.classes)

        # Create mapping from class name to index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to INPUT_DIR (e.g., "train_images/abc.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Process Targets
        if self.output_label:
            target = torch.zeros(self.num_classes, dtype=torch.float32)

            # Labels are space-delimited in the CSV
            if pd.notna(row["labels"]) and row["labels"] != "":
                labels_list = row["labels"].split()
                for label in labels_list:
                    if label in self.class_to_idx:
                        idx = self.class_to_idx[label]
                        target[idx] = 1.0
                    else:
                        # Handle unknown labels if necessary, or ignore
                        pass

            return image, target
        else:
            return image
