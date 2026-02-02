import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def get_transforms(image_size, phase="train"):
    """
    Returns the Albumentations transform pipeline based on the phase.

    Args:
        image_size (int): The target resolution (e.g., 380 or 224).
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                # Strong Geometric Augmentation as per Idea
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),  # Cite solution_lesson_node_00002
                # Note: Occlusion augmentations (Cutout, CoarseDropout) are strictly excluded.
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def prepare_folds(load_cached_data=True):
    """
    Loads metadata, merges train/val sets, creates stratified folds,
    and caches the result to parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with 'fold' column.
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORK_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to re-computing

    # 2. Compute from scratch
    # Load metadata files
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(
            "Metadata CSVs not found. Please ensure metadata is generated."
        )

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Merge to perform 5-Fold Stratified CV as per Idea
    df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Ensure stratify label exists
    if "stratify_label" not in df.columns:
        # Fallback logic if metadata didn't have it (though it should)
        label_cols = Config.CLASSES
        df["stratify_label"] = df[label_cols].idxmax(axis=1)

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["stratify_label"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    """
    Dataset class for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, output_label=True, debug=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            transforms (albumentations.Compose): Transforms to apply.
            output_label (bool): Whether to return labels (True for train/val, False for test).
            debug (bool): If True, limits dataset size for debugging.
        """
        super().__init__()
        self.df = df.copy()
        if debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        self.transforms = transforms
        self.output_label = output_label
        self.file_paths = self.df["file_path"].values
        self.image_ids = self.df["image_id"].values

        if self.output_label:
            self.labels = self.df[Config.CLASSES].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Construct full path
        rel_path = self.file_paths[index]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load Image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            image = self.transforms(image=image)["image"]

        # Return Logic
        if self.output_label:
            label = self.labels[index]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # For inference, return image_id to map predictions
            return image, self.image_ids[index]


def get_loaders(
    fold, image_size, batch_size, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold (int): The fold index to use for validation (0 to N_FOLDS-1).
        image_size (int): Image resolution.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        debug (bool): Debug mode flag.

    Returns:
        train_loader, val_loader
    """
    # Load processed dataframe with folds
    df = prepare_folds(load_cached_data=True)

    # Split
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df,
        transforms=get_transforms(image_size, phase="train"),
        output_label=True,
        debug=debug,
    )

    val_dataset = AppleDataset(
        val_df,
        transforms=get_transforms(image_size, phase="valid"),
        output_label=True,
        debug=debug,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(image_size, batch_size, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoader for the test set.
    """
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError("Test metadata not found.")

    test_df = pd.read_csv(Config.TEST_CSV)

    test_dataset = AppleDataset(
        test_df,
        transforms=get_transforms(image_size, phase="test"),
        output_label=False,
        debug=False,  # Always process full test set
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
