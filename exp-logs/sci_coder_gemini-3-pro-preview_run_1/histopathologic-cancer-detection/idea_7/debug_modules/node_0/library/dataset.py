import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed


class TumorDataset(Dataset):
    """
    PyTorch Dataset for the Pathology Classification Task.
    Reads images from disk, applies transformations, and returns tensors.
    """

    def __init__(self, df, transform=None, phase="train"):
        self.df = df
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        # cv2.imread loads in BGR format
        image = cv2.imread(file_path)

        # Handle potential loading failure
        if image is None:
            # Return a blank image to prevent crash, though this should be logged in a real scenario
            image = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare output dictionary
        data = {"image": image, "id": row["id"]}

        # Include label if available (Train/Val)
        if "label" in row:
            data["target"] = torch.tensor(row["label"], dtype=torch.float32)
        else:
            # Placeholder for Test set
            data["target"] = torch.tensor(-1.0, dtype=torch.float32)

        return data


def get_transforms(phase):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Common transforms
    # Hard Attention: Crop center 48x48 from the 96x96 original
    crop = A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE)
    norm = A.Normalize(mean=mean, std=std)
    to_tensor = ToTensorV2()

    if phase == "train":
        return A.Compose(
            [
                crop,
                # Augmentations as per Idea 7
                A.HorizontalFlip(p=Config.AUG_PROB),
                A.VerticalFlip(p=Config.AUG_PROB),
                A.RandomRotate90(p=Config.AUG_PROB),
                # Conservative color augmentation (No Hue/Saturation)
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS_LIMIT,
                    contrast_limit=Config.AUG_CONTRAST_LIMIT,
                    p=Config.AUG_PROB,
                ),
                norm,
                to_tensor,
            ]
        )
    else:
        # Validation and Test: Deterministic
        return A.Compose([crop, norm, to_tensor])


def get_folds(load_cached_data=True):
    """
    Generates or loads the stratified folds for Cross-Validation.

    Logic:
    1. Check if cached folds exist in working directory.
    2. If yes and load_cached_data is True, load and return.
    3. If no, load train/val metadata, combine them, create stratified folds.
    4. Save to cache and return.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Generate from scratch
    print("Generating new stratified folds...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    train_subset = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_subset = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine to get full training set
    full_df = pd.concat([train_subset, val_subset], axis=0).reset_index(drop=True)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Assign folds
    full_df["fold"] = -1
    for fold_idx, (_, val_indices) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_indices, "fold"] = fold_idx

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return full_df


def get_loaders(fold_idx, debug=False, load_cached_data=True):
    """
    Creates DataLoaders for the specified fold.

    Args:
        fold_idx (int): The fold index (0 to NUM_FOLDS-1) to use for validation.
        debug (bool): If True, subsamples the dataset for rapid testing.
        load_cached_data (bool): Whether to use cached fold definitions.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Get Fold Data
    df = get_folds(load_cached_data=load_cached_data)

    # Debugging: Subsample if requested
    if debug or Config.DEBUG_SAMPLE_SIZE is not None:
        sample_n = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG_SAMPLE_SIZE else 2000
        print(f"Debug mode: Subsampling dataset to {sample_n} samples.")
        df = df.sample(n=min(len(df), sample_n), random_state=Config.SEED).reset_index(
            drop=True
        )

    # Split into Train and Validation based on fold_idx
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # 2. Get Test Data
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug or Config.DEBUG_SAMPLE_SIZE is not None:
        sample_n = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG_SAMPLE_SIZE else 2000
        test_df = test_df.sample(
            n=min(len(test_df), sample_n), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Create Datasets
    train_dataset = TumorDataset(
        train_df, transform=get_transforms("train"), phase="train"
    )
    val_dataset = TumorDataset(val_df, transform=get_transforms("valid"), phase="valid")
    test_dataset = TumorDataset(test_df, transform=get_transforms("test"), phase="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last to maintain batch statistics
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

    return train_loader, val_loader, test_loader
