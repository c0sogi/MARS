import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratifiedKFold
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed


def get_folds(load_cached_data=True):
    """
    Generates or loads the 5-fold cross-validation split for the development dataset.

    Args:
        load_cached_data (bool): If True, attempts to load folds from a cached Parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the combined train/val data with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading folds from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating new folds...")

    # 2. Load and combine metadata
    # We combine the provided 'train' and 'val' splits into a single 'dev' set
    # because we want to perform our own 5-fold CV as per the strategy.
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError("Metadata CSV files not found.")

    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)

    # Combine
    df_dev = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # 3. Prepare for Iterative Stratification
    # Identify label columns
    label_cols = [c for c in df_dev.columns if c.startswith("species_")]
    # Sort to ensure consistent order
    label_cols.sort(key=lambda x: int(x.split("_")[1]))

    X = df_dev["rec_id"].values.reshape(-1, 1)
    y = df_dev[label_cols].values

    # 4. Perform Split
    # Seed the splitter
    k_fold = IterativeStratifiedKFold(
        n_splits=Config.N_FOLDS, order=1, random_state=Config.SEED
    )

    df_dev["fold"] = -1

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df_dev.loc[val_indices, "fold"] = fold_idx

    # 5. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_dev.to_parquet(cache_path, index=False)
    print(f"Folds saved to {cache_path}")

    return df_dev


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles loading spectrograms, preprocessing, and augmentation.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filenames, labels).
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label loading.
            transform (A.Compose): Albumentations transforms to apply.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Identify label columns
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]
        self.label_cols.sort(key=lambda x: int(x.split("_")[1]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # Metadata contains 'file_path_spec' (e.g., supplemental_data/spectrograms/PC10...bmp)
        # We need to map this to the Filtered Spectrograms directory defined in Config.
        original_rel_path = row["file_path_spec"]
        filename = os.path.basename(original_rel_path)
        img_path = os.path.join(Config.IMAGE_DIR, filename)

        # Load as grayscale (spectrograms are single channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should be caught by metadata check, but for safety)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 2. Pseudo-RGB
        # Replicate channel to create 3-channel image
        image = cv2.merge([image, image, image])

        # 3. Apply Albumentations (Resize, Normalize, etc.)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            augmented = base_transform(image=image)
            image = augmented["image"]

        # 4. Apply SpecAugment (Training Only)
        # Applied on the tensor: (C, H, W)
        if self.mode == "train":
            image = self._apply_spec_augment(image)

        # 5. Get Labels
        if self.mode in ["train", "val"]:
            labels = row[self.label_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # For test set, return image and a dummy label or ID
            # Returning ID helps with submission creation if needed,
            # but standard usually returns image, label (or empty)
            # We will return dummy zeros for consistency
            dummy_labels = np.zeros(len(self.label_cols), dtype=np.float32)
            return image, torch.tensor(dummy_labels)

    def _apply_spec_augment(self, img_tensor):
        """
        Applies Time and Frequency masking to the tensor.
        img_tensor: (C, H, W)
        """
        # Clone to avoid modifying in place if referenced elsewhere
        aug_img = img_tensor.clone()
        C, H, W = aug_img.shape

        # Frequency Masking (H axis)
        # Mask a strip of height F
        F = Config.FREQ_MASK_PARAM
        f = np.random.randint(0, F + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, H - f))
            aug_img[:, f0 : f0 + f, :] = 0

        # Time Masking (W axis)
        # Mask a strip of width T
        T = Config.TIME_MASK_PARAM
        t = np.random.randint(0, T + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, W - t))
            aug_img[:, :, t0 : t0 + t] = 0

        return aug_img


class MixupCollate:
    """
    Collate function that applies Mixup augmentation to a batch.
    """

    def __init__(self, alpha=Config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, label)
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        # Only apply mixup if alpha > 0
        if self.alpha > 0:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Create shuffled indices
            batch_size = images.size(0)
            index = torch.randperm(batch_size)

            # Mix images
            mixed_images = lam * images + (1 - lam) * images[index, :]

            # Mix labels
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            return mixed_images, mixed_labels

        return images, labels


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
