import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_spectrogram_path(wav_rel_path):
    """
    Converts a relative WAV path (from metadata) to the absolute Spectrogram BMP path.
    Example: 'essential_data/src_wavs/file.wav' -> './input/supplemental_data/spectrograms/file.bmp'
    """
    filename = os.path.basename(wav_rel_path)
    bmp_filename = os.path.splitext(filename)[0] + ".bmp"
    return os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for training or inference.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMAGE_SIZE[0] // 8,
                    max_width=Config.IMAGE_SIZE[1] // 8,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    def __init__(self, df, transforms=None, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test

        # Pre-compute spectrogram paths
        self.paths = [get_spectrogram_path(p) for p in df["file_path"].values]

        # Identify label columns
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Load labels
        if not self.is_test:
            # Ensure float32 for BCEWithLogitsLoss
            # Handles both binary ints (ground truth) and floats (pseudo-labels)
            self.labels = df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(df), Config.NUM_CLASSES), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Load Image as Grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Safety fallback, though EDA suggests data is clean
            img = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

        # Channel Replication: Grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        label = self.labels[idx]

        return img, torch.tensor(label, dtype=torch.float32)


def load_and_process_metadata(
    debug=False, pseudo_labels_path=None, load_cached_data=True
):
    """
    Loads metadata, optionally merges pseudo-labels, and caches the result.
    Strictly follows the caching logic requirement.
    """
    # Construct cache filename based on arguments
    cache_name = "metadata_cache"
    if debug:
        cache_name += "_debug"
    if pseudo_labels_path:
        cache_name += "_pseudo"

    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            full_df = pd.read_parquet(cache_path)

            # Split back into train, val, test
            train_df = full_df[full_df["split_name"] == "train"].drop(
                columns=["split_name"]
            )
            val_df = full_df[full_df["split_name"] == "val"].drop(
                columns=["split_name"]
            )
            test_df = full_df[full_df["split_name"] == "test"].drop(
                columns=["split_name"]
            )

            return train_df, val_df, test_df
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing from scratch...")

    # 2. Compute from scratch
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply Debug Subset
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Merge Pseudo-Labels if provided
    if pseudo_labels_path and os.path.exists(pseudo_labels_path):
        try:
            # Assume pseudo-labels are in wide format (rec_id, species_0, ...) or similar
            if pseudo_labels_path.endswith(".parquet"):
                pseudo_df = pd.read_parquet(pseudo_labels_path)
            else:
                pseudo_df = pd.read_csv(pseudo_labels_path)

            # We need to map pseudo-labels (which have rec_id) to file paths.
            # The file paths for these rec_ids are in test_df.
            test_meta = test_df[["rec_id", "file_path"]]

            # Merge to get file paths for the pseudo-labeled data
            merged_pseudo = pd.merge(test_meta, pseudo_df, on="rec_id", how="inner")

            # Ensure all label columns exist
            label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
            if all(c in merged_pseudo.columns for c in label_cols):
                # Add empty labels_str column to match train schema (optional but good for consistency)
                merged_pseudo["labels_str"] = ""

                # Align columns with train_df
                common_cols = [
                    c for c in train_df.columns if c in merged_pseudo.columns
                ]
                merged_pseudo = merged_pseudo[common_cols]

                # Concatenate: Train + Pseudo
                train_df = pd.concat(
                    [train_df, merged_pseudo], axis=0, ignore_index=True
                )
            else:
                print(
                    "Warning: Pseudo-label file missing required species columns. Skipping merge."
                )

        except Exception as e:
            print(f"Error merging pseudo-labels: {e}")

    # 3. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Tag rows to reconstruct splits later
    train_df["split_name"] = "train"
    val_df["split_name"] = "val"
    test_df["split_name"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    full_df.to_parquet(cache_path, index=False)

    # Clean up before returning
    train_df = train_df.drop(columns=["split_name"])
    val_df = val_df.drop(columns=["split_name"])
    test_df = test_df.drop(columns=["split_name"])

    return train_df, val_df, test_df


def get_dataloaders(debug=False, pseudo_labels_path=None, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Metadata
    train_df, val_df, test_df = load_and_process_metadata(
        debug=debug,
        pseudo_labels_path=pseudo_labels_path,
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, transforms=get_transforms("train"), is_test=False
    )
    val_dataset = BirdDataset(val_df, transforms=get_transforms("val"), is_test=False)
    test_dataset = BirdDataset(test_df, transforms=get_transforms("val"), is_test=True)

    # Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
