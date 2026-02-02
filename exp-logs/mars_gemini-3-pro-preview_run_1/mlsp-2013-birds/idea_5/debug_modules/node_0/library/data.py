import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Normalize to standard ImageNet mean/std
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # SpecAugment-like masking using CoarseDropout
                # We use multiple holes to simulate time/freq masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.15),
                    max_width=int(Config.IMG_WIDTH * 0.15),
                    min_holes=2,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_and_process_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads metadata, processes images (load, resize, stack), and returns arrays.
    Implements caching using .npy files in the working directory.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        cache_prefix (str): Prefix for the cache filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, rec_ids)
            images: np.ndarray of shape (N, H, W, 3)
            labels: np.ndarray of shape (N, NumClasses)
            rec_ids: np.ndarray of shape (N,)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            try:
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                rec_ids = np.load(ids_cache_path)
                print(f"Loaded {cache_prefix} data from cache: {len(images)} samples.")
                return images, labels, rec_ids
            except Exception as e:
                print(f"Failed to load cache for {cache_prefix}: {e}. Recomputing...")
        else:
            print(f"Cache not found for {cache_prefix}. Processing from scratch...")

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Debugging: limit size
    if Config.DEBUG_SAMPLE_SIZE is not None:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    img_list = []
    label_list = []
    id_list = []

    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    print(f"Processing {len(df)} samples for {cache_prefix}...")

    for idx, row in df.iterrows():
        # Construct path to spectrogram
        # row['file_path'] is like 'essential_data/src_wavs/PC10_...wav'
        wav_rel_path = row["file_path"]
        filename = os.path.basename(wav_rel_path)
        bmp_name = filename.replace(".wav", ".bmp")
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)

        rec_id = row["rec_id"]

        # Load Labels
        # If columns exist, load them. Else (e.g. if metadata format changes), use 0s.
        # The metadata provided definitely has species_X columns.
        current_labels = row[label_cols].values.astype(np.float32)

        # Load Image
        if os.path.exists(bmp_path):
            # Load as grayscale (unchanged)
            img = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                # Fallback: create black image
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
            else:
                # Resize to Densified dimensions
                img = cv2.resize(
                    img,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Convert to 3 channels (Replicate)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            # If already 3 channels (unlikely for these BMPs but possible), keep it.

            img_list.append(img)
            label_list.append(current_labels)
            id_list.append(rec_id)
        else:
            # Skip missing files or handle them?
            # Given dataset size is small, skipping might be dangerous if many are missing.
            # But we verified in metadata step that files exist.
            # We'll insert a zero image to keep alignment with metadata if critical,
            # but appending only valid ones is safer for training.
            print(f"Warning: Image not found {bmp_path}")
            continue

    images = np.array(img_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.float32)
    rec_ids = np.array(id_list, dtype=np.int64)

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(ids_cache_path, rec_ids)

    print(f"Processed and cached {len(images)} samples for {cache_prefix}.")

    return images, labels, rec_ids


class BirdDataset(Dataset):
    def __init__(self, images, labels, transforms=None, soft_labels=None):
        """
        Args:
            images (np.ndarray): Shape (N, H, W, 3)
            labels (np.ndarray): Shape (N, NumClasses) - Hard labels
            transforms (A.Compose): Albumentations transforms
            soft_labels (np.ndarray, optional): Shape (N, NumClasses).
                                                If provided, these override the hard labels.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Determine label to return
        if self.soft_labels is not None:
            label = self.soft_labels[idx]
        else:
            label = self.labels[idx]

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, torch.tensor(label, dtype=torch.float32)


def create_loader(
    images,
    labels,
    batch_size,
    shuffle=False,
    transform_mode="val",
    num_workers=Config.NUM_WORKERS,
):
    """
    Helper to create a DataLoader from arrays.
    Useful for creating the combined Student loader.
    """
    transforms = get_transforms(transform_mode)
    dataset = BirdDataset(images, labels, transforms=transforms)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )
    return loader


def get_dataloaders(load_cached_data=True, return_arrays=False):
    """
    Main entry point to get standard Train, Val, and Test loaders.

    Args:
        load_cached_data (bool): Use caching.
        return_arrays (bool): If True, returns the raw numpy arrays instead of loaders.
                              (Useful for constructing the Student dataset).

    Returns:
        If return_arrays=False:
            (train_loader, val_loader, test_loader, test_ids)
        If return_arrays=True:
            (train_data, val_data, test_data) where each is (images, labels, ids)
    """
    # Load Data
    train_imgs, train_lbls, train_ids = load_and_process_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_and_process_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_and_process_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    if return_arrays:
        return (
            (train_imgs, train_lbls, train_ids),
            (val_imgs, val_lbls, val_ids),
            (test_imgs, test_lbls, test_ids),
        )

    # Create Datasets
    train_ds = BirdDataset(train_imgs, train_lbls, transforms=get_transforms("train"))
    val_ds = BirdDataset(val_imgs, val_lbls, transforms=get_transforms("val"))
    test_ds = BirdDataset(
        test_imgs, test_lbls, transforms=get_transforms("val")
    )  # Test uses val transforms (no aug)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
