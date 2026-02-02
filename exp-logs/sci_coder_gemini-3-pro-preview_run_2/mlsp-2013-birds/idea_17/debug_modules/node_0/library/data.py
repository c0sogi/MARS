import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratifiedKFold

from library import config


def get_folds(load_cached_data=True):
    """
    Generates or loads the 5-fold iterative stratified split for the training data.

    Args:
        load_cached_data (bool): If True, attempts to load folds from disk.

    Returns:
        pd.DataFrame: The training dataframe with an added 'fold' column.
    """
    cache_path = os.path.join(config.WORKING_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded folds from {cache_path}")
            return df
        except Exception as e:
            # print(f"Failed to load cache: {e}. Recomputing...")
            pass

    # 2. Compute from scratch
    df = pd.read_csv(config.TRAIN_CSV)

    # Extract labels for stratification
    label_cols = [c for c in df.columns if c.startswith("species_")]
    X = df["rec_id"].values.reshape(-1, 1)
    y = df[label_cols].values

    # Initialize folds
    df["fold"] = -1
    mskf = IterativeStratifiedKFold(n_splits=config.N_FOLDS, order=1)

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        df.loc[val_idx, "fold"] = fold_idx

    # 3. Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles Filtered Spectrograms, Pseudo-RGB conversion, Time-Rolling, and Soft Labels.
    """

    def __init__(self, df, transforms=None, soft_labels=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (rec_id, file_path_spec, labels).
            transforms (A.Compose): Albumentations transforms pipeline.
            soft_labels (dict, optional): Dictionary mapping rec_id to soft label arrays (Stage 2).
            mode (str): 'train', 'val', or 'test'. Controls augmentation application.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.soft_labels = soft_labels
        self.mode = mode

        # Identify label columns
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]

        # Pre-compute paths to avoid overhead in __getitem__
        # We use the filename from the metadata but point to the FILTERED spectrogram directory
        self.image_paths = []
        for _, row in self.df.iterrows():
            # Extract filename (e.g., PC10_...bmp) from the relative path provided in metadata
            fname = os.path.basename(row["file_path_spec"])
            full_path = os.path.join(config.SPECTROGRAM_DIR, fname)
            self.image_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]
        img_path = self.image_paths[idx]

        # 1. Load Image
        # Load as grayscale (H, W). If fails, create a black image.
        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = None

        if image is None:
            # Fallback for missing files (should be rare based on EDA)
            # Default size 256x512 is arbitrary, will be resized by transforms
            image = np.zeros((256, 512), dtype=np.uint8)

        # 2. Time-Rolling Augmentation (Train only)
        # Shift the spectrogram along the time axis (axis 1)
        if self.mode == "train":
            # Random shift
            shift = np.random.randint(0, image.shape[1])
            image = np.roll(image, shift, axis=1)

        # 3. Pseudo-RGB Conversion
        # Stack 3 channels to match ImageNet pretrained weights
        image = cv2.merge([image, image, image])

        # 4. Apply Transforms (Resize, Normalize, ToTensor)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 5. Get Targets
        # Hard Labels (Ground Truth)
        if len(self.label_cols) > 0:
            target = row[self.label_cols].values.astype(np.float32)
        else:
            # Test set might not have label columns or they might be placeholders
            target = np.zeros(config.NUM_SPECIES, dtype=np.float32)

        # Soft Labels (Teacher Distillation)
        soft_target = np.zeros(config.NUM_SPECIES, dtype=np.float32)
        if self.soft_labels is not None and rec_id in self.soft_labels:
            soft_target = self.soft_labels[rec_id].astype(np.float32)

        return {
            "image": image,
            "target": torch.tensor(target),
            "soft_target": torch.tensor(soft_target),
            "rec_id": rec_id,
        }


def get_transforms(model_name, mode="train"):
    """
    Creates the Albumentations transform pipeline based on model config and mode.

    Args:
        model_name (str): Name of the backbone (e.g., 'resnet18', 'densenet121').
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Retrieve size from config
    if model_name in config.MODEL_CONFIGS:
        height, width = config.MODEL_CONFIGS[model_name]["img_size"]
    else:
        # Fallback
        height, width = 224, 448

    # ImageNet Normalization Constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                # Note: Time-Rolling is handled in Dataset.__getitem__
                # Note: Mixup is handled in the training loop
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


def create_loaders(
    fold, model_name, batch_size=config.BATCH_SIZE, soft_labels=None, debug=False
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold (int): The fold index to use for validation (0-4).
        model_name (str): Backbone name for transform configuration.
        batch_size (int): Batch size.
        soft_labels (dict, optional): Soft labels for distillation (Stage 2).
        debug (bool): If True, truncates dataset for quick testing.

    Returns:
        train_loader, val_loader
    """
    # 1. Get Data with Folds
    df = get_folds(load_cached_data=True)

    # 2. Split Train/Val
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Debugging: Truncate if requested
    if debug:
        limit = config.MAX_DEBUG_SAMPLES if config.MAX_DEBUG_SAMPLES else 50
        train_df = train_df.head(limit)
        val_df = val_df.head(limit)
    elif config.MAX_DEBUG_SAMPLES is not None:
        train_df = train_df.head(config.MAX_DEBUG_SAMPLES)
        val_df = val_df.head(config.MAX_DEBUG_SAMPLES)

    # 3. Create Datasets
    train_dataset = BirdDataset(
        train_df,
        transforms=get_transforms(model_name, mode="train"),
        soft_labels=soft_labels,
        mode="train",
    )

    val_dataset = BirdDataset(
        val_df,
        transforms=get_transforms(model_name, mode="val"),
        soft_labels=None,  # No soft labels needed for validation
        mode="val",
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def create_test_loader(model_name, batch_size=config.BATCH_SIZE):
    """
    Creates a DataLoader for the test set.
    """
    df_test = pd.read_csv(config.TEST_CSV)

    dataset = BirdDataset(
        df_test, transforms=get_transforms(model_name, mode="test"), mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
