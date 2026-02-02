import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything


def load_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Loads images from disk, converts to RGB, and caches them as a numpy array.
    If a cache file exists and load_cached_data is True, loads from cache instead.
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    suffix = "_debug" if Config.DEBUG else ""
    images_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images{suffix}.npy")
    labels_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels{suffix}.npy")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(labels_cache_path)
    ):
        print(f"Loading {cache_prefix} data from cache: {images_cache_path}")
        try:
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            return images, labels
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source
    print(f"Loading {cache_prefix} data from source images...")
    image_list = []
    label_list = []

    # Pre-allocate for speed if possible, but list append is robust for variable length if any issues
    # Given fixed size, we could pre-allocate, but list append is fine for 175k in modern python

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {cache_prefix}"):
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        label = row["label"]

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            # In case of missing image, we might skip or fill black.
            # Based on metadata check, all files exist.
            print(f"Warning: Image not found at {file_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_list.append(img)
        label_list.append(label)

    images = np.array(image_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.int64)

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)

    return images, labels


class MemoryPathologyDataset(Dataset):
    """
    Dataset that holds all images in RAM as a numpy array.
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return float tensor for label (BCEWithLogitsLoss expects float)
        # Or Long tensor if using CrossEntropy. Config says NUM_CLASSES=1, usually implies BCE.
        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    Implements 'Augment-then-Crop' strategy.
    """
    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations on full 96x96 image
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Continuous Random Rotation (-180 to 180)
                A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
                # Color Augmentations
                # "I will apply ColorJitter to 100% of the training images"
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                ),
                # Contextual Crop to 64x64
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                # Normalization
                A.Normalize(mean=Config.DATASET_MEAN, std=Config.DATASET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                A.Normalize(mean=Config.DATASET_MEAN, std=Config.DATASET_STD),
                ToTensorV2(),
            ]
        )


def get_loaders(fold, seed, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in the Bagged Ensemble strategy.
    Combines train and val metadata, then splits using StratifiedKFold.
    """
    seed_everything(seed)

    # 1. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        df_train_meta = df_train_meta.head(Config.DEBUG_SAMPLE_SIZE)
        df_val_meta = df_val_meta.head(Config.DEBUG_SAMPLE_SIZE)
        print(
            f"[DEBUG] Reduced dataset to {len(df_train_meta)} train and {len(df_val_meta)} val samples."
        )

    # 2. Load Images (Cached)
    # We load them separately to match the metadata files, then concatenate
    train_imgs, train_lbls = load_and_cache_data(
        df_train_meta, "train", load_cached_data
    )
    val_imgs, val_lbls = load_and_cache_data(df_val_meta, "val", load_cached_data)

    # 3. Concatenate for Cross-Validation
    all_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    all_lbls = np.concatenate([train_lbls, val_lbls], axis=0)

    # 4. Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=seed)

    # Get indices for the specific fold
    # skf.split returns a generator, we iterate to the requested fold
    splits = list(skf.split(all_imgs, all_lbls))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {Config.NUM_FOLDS} splits.")

    train_idx, val_idx = splits[fold]

    # 5. Create Datasets
    train_ds = MemoryPathologyDataset(
        all_imgs[train_idx], all_lbls[train_idx], transform=get_transforms("train")
    )
    val_ds = MemoryPathologyDataset(
        all_imgs[val_idx], all_lbls[val_idx], transform=get_transforms("val")
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Load test images
    test_imgs, _ = load_and_cache_data(df_test, "test", load_cached_data)
    # Create dummy labels for dataset compatibility
    test_lbls = np.zeros(len(test_imgs), dtype=np.int64)

    test_ds = MemoryPathologyDataset(
        test_imgs, test_lbls, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return test_loader, df_test["id"].values
