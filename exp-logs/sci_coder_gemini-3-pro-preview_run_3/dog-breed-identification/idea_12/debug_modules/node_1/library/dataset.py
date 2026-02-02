import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a sorted list of unique classes (breeds) from the training metadata.
    Uses caching to ensure determinism across runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        list: Sorted list of breed names.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "classes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_classes = pd.read_parquet(cache_path)
            return df_classes["breed"].tolist()
        except Exception as e:
            print(f"Failed to load class cache: {e}. Regenerating...")

    # 2. Generate from scratch
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    unique_breeds = sorted(df_train["breed"].unique().tolist())

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_classes = pd.DataFrame({"breed": unique_breeds})
    df_classes.to_parquet(cache_path, index=False)

    return unique_breeds


def get_transforms(phase):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # 1. Random Resized Crop
                A.RandomResizedCrop(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333),
                    p=1.0,
                ),
                # 2. Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # 3. RandAugment Approximation
                # We select 2 transformations from a set of geometric and color augmentations
                A.SomeOf(
                    [
                        A.Affine(rotate=(-30, 30), p=0.5),
                        A.Affine(shear=(-10, 10), p=0.5),
                        A.Affine(translate_percent=(-0.1, 0.1), p=0.5),
                        A.Solarize(p=0.5),
                        A.Posterize(p=0.5),
                        A.Equalize(p=0.5),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                        ),
                        A.Sharpen(p=0.5),
                    ],
                    n=2,
                    p=1.0,
                ),
                # Normalization and Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test: Deterministic Resize and Center Crop
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=256, p=1.0),
                A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    """

    def __init__(self, df, class_list, transform=None, input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and optionally 'breed'.
            class_list (list): List of unique breed names to map labels to integers.
            transform (A.Compose, optional): Albumentations transform pipeline.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.input_dir = input_dir

        # Create mapping from breed name to index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_list)}

        # Pre-check if labels exist
        self.has_labels = "breed" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like 'train/id.jpg'
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for robustness, though metadata validation should prevent this
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(size=(Config.IMG_SIZE, Config.IMG_SIZE)),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # Get label
        label = -1
        if self.has_labels:
            breed_name = row["breed"]
            label = self.class_to_idx.get(breed_name, -1)

        return image, label


def load_metadata():
    """
    Loads train, val, and test metadata dataframes from the generated CSVs.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    return df_train, df_val, df_test
