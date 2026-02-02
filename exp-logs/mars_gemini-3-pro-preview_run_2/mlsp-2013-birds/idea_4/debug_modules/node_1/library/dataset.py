import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config


# --- SpecAugment Implementation ---
class SpecAugment:
    """
    Applies SpecAugment (Time and Frequency Masking) to an image.
    Simulates partial signal loss common in audio recordings.
    """

    def __init__(
        self,
        num_mask=2,
        freq_masking_max_percentage=0.10,
        time_masking_max_percentage=0.10,
        always_apply=False,
        p=0.5,
    ):
        self.num_mask = num_mask
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.time_masking_max_percentage = time_masking_max_percentage
        self.always_apply = always_apply
        self.p = p

    def __call__(self, image, **kwargs):
        if not self.always_apply and np.random.rand() > self.p:
            return image

        aug_image = image.copy()
        h, w = aug_image.shape[:2]

        for _ in range(self.num_mask):
            # Frequency Masking (Horizontal strips in spectrogram)
            freq_max_h = int(self.freq_masking_max_percentage * h)
            f = np.random.randint(0, max(1, freq_max_h))
            f0 = np.random.randint(0, max(1, h - f))
            aug_image[f0 : f0 + f, :] = 0

            # Time Masking (Vertical strips in spectrogram)
            time_max_w = int(self.time_masking_max_percentage * w)
            t = np.random.randint(0, max(1, time_max_w))
            t0 = np.random.randint(0, max(1, w - t))
            aug_image[:, t0 : t0 + t] = 0

        return aug_image


# --- Dataset Class ---
class BirdDataset(Dataset):
    def __init__(self, df, config, transform=None, is_train=False):
        self.df = df
        self.config = config
        self.transform = transform
        self.is_train = is_train

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = df[self.label_cols].values.astype(np.float32)

        # Pre-calculate file paths
        self.file_paths = []
        for _, row in df.iterrows():
            # Extract filename from the metadata path
            # Metadata path example: supplemental_data/spectrograms/PC10_....bmp
            orig_path = row["file_path_spec"]
            basename = os.path.basename(orig_path)
            # Construct path to filtered spectrograms as per strategy
            full_path = os.path.join(self.config.spectrogram_dir, basename)
            self.file_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        labels = self.labels[idx]

        # Load Image
        # Load as BGR (OpenCV default)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if img is None:
            # Fallback for safety (create black image)
            img = np.zeros(
                (self.config.image_size[0], self.config.image_size[1], 3),
                dtype=np.uint8,
            )
        else:
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            # Albumentations pipeline
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, torch.tensor(labels)


# --- Transforms ---
def get_transforms(config, data_type="train"):
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(config.image_size[0], config.image_size[1]),
                # Photometric Distortions (Robustness to gain/recording quality)
                A.RandomBrightnessContrast(p=0.5),
                # SpecAugment via Lambda
                A.Lambda(
                    name="SpecAugment", image=SpecAugment(num_mask=2, p=0.5), p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.image_size[0], config.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


# --- Stratified Folds Generation ---
def get_stratified_folds(config, load_cached_data=True):
    """
    Merges train and val metadata, creates stratified K-Folds,
    and returns the dataframe with a 'fold' column.
    Caches the result to disk to ensure deterministic processing.
    """
    cache_path = os.path.join(config.working_dir, "folds_data.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating stratified folds...")
    # Load original splits provided by metadata script
    train_df = pd.read_csv(config.train_csv_path)
    val_df = pd.read_csv(config.val_csv_path)

    # Merge them into a single development set
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare for Iterative Stratification
    label_cols = [c for c in df.columns if c.startswith("species_")]
    y = df[label_cols].values
    X = df["rec_id"].values.reshape(-1, 1)  # Dummy X required by sklearn API

    # Initialize folds
    df["fold"] = -1

    # Create folds
    kfold = IterativeStratification(n_splits=config.n_folds, order=1)

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
        df.loc[val_indices, "fold"] = fold_idx

    # Save to cache
    os.makedirs(config.working_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return df


# --- DataLoader Creation ---
def create_dataloaders(config, fold_idx=0):
    """
    Creates train and validation dataloaders for a specific fold.
    """
    # Get data with folds
    df = get_stratified_folds(config, load_cached_data=True)

    # Split based on fold_idx
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Transforms
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")

    # Datasets
    train_dataset = BirdDataset(
        train_df, config, transform=train_transform, is_train=True
    )
    val_dataset = BirdDataset(val_df, config, transform=val_transform, is_train=False)

    # Reproducibility settings for DataLoader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,  # Stabilize BatchNorm statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_test_dataloader(config):
    """
    Creates dataloader for the test set.
    """
    if not os.path.exists(config.test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {config.test_csv_path}")

    df_test = pd.read_csv(config.test_csv_path)
    test_transform = get_transforms(config, "val")  # No augmentation for test

    test_dataset = BirdDataset(
        df_test, config, transform=test_transform, is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return test_loader
