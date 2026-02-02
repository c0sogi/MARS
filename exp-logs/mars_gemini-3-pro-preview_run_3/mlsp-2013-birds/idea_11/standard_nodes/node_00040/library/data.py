import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import parse_label_string


def get_transforms(phase: str):
    """
    Returns the augmentation pipeline for the specified phase.

    Args:
        phase (str): 'train' or 'valid' (also used for test).

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                # Horizontal Translation (Time-Shift) only.
                # Avoid vertical shift (frequency shift) and rotation.
                # Cite solution_lesson_node_00007.
                A.Affine(
                    translate_percent={"x": 0.1, "y": 0.0},
                    scale=1.0,
                    rotate=0,
                    shear=0,
                    cval=0,
                    mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Loads BMP spectrograms, converts to 3-channel pseudo-RGB, and applies transforms.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, labels).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct path to BMP spectrogram
        # Metadata file_path points to wav: essential_data/src_wavs/filename.wav
        # We need: supplemental_data/spectrograms/filename.bmp
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = wav_filename.replace(".wav", ".bmp")
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        # Load as grayscale first
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files (though verification passed, good for robustness)
        if image is None:
            # Create a blank black image of expected size if load fails
            # Original spectrograms are roughly 500x129 or similar depending on FFT
            # We'll create a placeholder that will get resized
            image = np.zeros((128, 500), dtype=np.uint8)

        # 3-Channel Rule: Replicate grayscale to 3 channels
        image = np.stack([image, image, image], axis=-1)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Labels
        # parse_label_string handles '?' and converts to binary vector
        labels = parse_label_string(row["labels"])
        labels = torch.tensor(labels, dtype=torch.float32)

        rec_id = row["rec_id"]

        return image, labels, rec_id


def make_folds(load_cached_data=True):
    """
    Creates 5 folds using Iterative Stratification.
    Merges train.csv and val.csv from metadata to form the full dev set.

    Args:
        load_cached_data (bool): If True, tries to load from cache.

    Returns:
        pd.DataFrame: DataFrame with 'fold' column updated/added.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading folds from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Creating folds...")
    # Load metadata
    train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_path = os.path.join(Config.METADATA_DIR, "val.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Merge to create full development set
    df = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # Prepare X and y for stratification
    # X can be just indices
    X = df.index.values.reshape(-1, 1)

    # Create binary label matrix for stratification
    y_list = []
    for labels_str in df["labels"]:
        y_list.append(parse_label_string(labels_str))
    y = np.array(y_list)

    # Initialize Iterative Stratification
    stratifier = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    # Assign folds
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(stratifier.split(X, y)):
        df.loc[val_idx, "fold"] = fold_idx

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Folds saved to {cache_path}")

    return df


def get_loaders(
    fold_idx, df_folds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): Index of the validation fold (0 to N_FOLDS-1).
        df_folds (pd.DataFrame): DataFrame containing the data and fold info.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.

    Returns:
        tuple: (train_loader, valid_loader)
    """
    train_df = df_folds[df_folds["fold"] != fold_idx].reset_index(drop=True)
    valid_df = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)

    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        valid_df = valid_df.head(Config.DEBUG_SUBSET_SIZE)

    train_dataset = BirdDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    valid_dataset = BirdDataset(
        valid_df, transforms=get_transforms("valid"), mode="valid"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    test_path = os.path.join(Config.METADATA_DIR, "test.csv")
    df_test = pd.read_csv(test_path)

    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    test_dataset = BirdDataset(
        df_test,
        transforms=get_transforms(
            "valid"
        ),  # Use valid transforms (deterministic) for test
        mode="test",
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
