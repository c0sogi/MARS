import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    Custom Dataset for Bird Species Classification.
    Handles loading of processed spectrograms and labels.
    Implements the structural innovation of Spectral Deltas.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N, NumClasses).
            ids (np.ndarray, optional): Array of record IDs (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are already processed into (H, W, 3) float32 in the caching step
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback conversion if no transform provided
            image = torch.tensor(image).permute(2, 0, 1)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        elif self.ids is not None:
            rec_id = self.ids[idx]
            return image, rec_id
        else:
            return image

    @staticmethod
    def compute_deltas(spec):
        """
        Computes the first and second temporal derivatives (deltas) of the spectrogram
        to create a 3-channel input.

        Args:
            spec (np.ndarray): Single channel spectrogram (H, W).

        Returns:
            np.ndarray: 3-channel image (H, W, 3) -> [Intensity, Delta, Delta-Delta].
        """
        # Ensure input is float
        spec = spec.astype(np.float32)

        # Channel 1: Intensity (Original)
        c1 = spec

        # Channel 2: Delta (First derivative along time axis / width)
        # axis 1 is width (time)
        c2 = np.gradient(spec, axis=1)

        # Channel 3: Delta-Delta (Second derivative)
        c3 = np.gradient(c2, axis=1)

        # Stack channels
        img_3c = np.stack([c1, c2, c3], axis=-1)

        return img_3c


def process_and_cache_data(
    metadata_path,
    cache_img_path,
    cache_label_path,
    cache_id_path=None,
    load_cached_data=True,
):
    """
    Loads raw data, processes it (Resize + Deltas), and caches it to disk.

    Args:
        metadata_path (str): Path to the CSV file (train/val/test).
        cache_img_path (str): Path to save/load cached images .npy.
        cache_label_path (str): Path to save/load cached labels .npy.
        cache_id_path (str, optional): Path to save/load cached IDs .npy.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_img_path) and os.path.exists(cache_label_path):
            # Check ID cache if required
            if cache_id_path is None or os.path.exists(cache_id_path):
                print(f"Loading cached data from {os.path.dirname(cache_img_path)}...")
                images = np.load(cache_img_path)
                labels = np.load(cache_label_path)
                ids = np.load(cache_id_path) if cache_id_path else None
                return images, labels, ids

    # 2. Process from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    # Identify label columns
    # Explicitly select columns based on config to avoid artifacts (Cite debug_lesson_1)
    target_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    label_cols = [c for c in target_cols if c in df.columns]

    for idx, row in df.iterrows():
        # Construct full path to spectrogram
        # row['file_path'] is like 'essential_data/src_wavs/filename.wav'
        # Spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = os.path.splitext(wav_filename)[0] + ".bmp"
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        if not os.path.exists(bmp_path):
            # Should not happen based on metadata verification, but safe to skip
            continue

        # Load Image
        img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize
        # cv2.resize expects (Width, Height)
        img_resized = cv2.resize(
            img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

        # Normalize to 0-1 range before delta computation
        img_norm = img_resized.astype(np.float32) / 255.0

        # Compute Deltas (Structural Innovation)
        img_3c = BirdDataset.compute_deltas(img_norm)

        img_list.append(img_3c)
        id_list.append(row["rec_id"])

        # Handle Labels
        if len(label_cols) > 0:
            lbls = row[label_cols].values.astype(np.float32)
            label_list.append(lbls)
        else:
            # Dummy labels for test set if columns missing (though metadata has them as 0)
            label_list.append(np.zeros(Config.NUM_CLASSES, dtype=np.float32))

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    ids = np.array(id_list, dtype=np.int64)

    # Save to Cache
    os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
    np.save(cache_img_path, images)
    np.save(cache_label_path, labels)
    if cache_id_path:
        np.save(cache_id_path, ids)

    return images, labels, ids


def get_transforms(stage="train"):
    """
    Returns Albumentations transforms for the specified stage.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if stage == "train":
        return A.Compose(
            [
                # SpecAugment-like masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_HEIGHT // 8,
                    max_width=Config.IMG_WIDTH // 8,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize and Convert
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def get_data_loaders(config, pseudo_labels=None, load_cached_data=True):
    """
    Prepares and returns DataLoaders for Train, Val, and Test sets.

    Args:
        config (Config): Configuration class.
        pseudo_labels (dict or np.ndarray, optional): Soft labels for the test set.
                                                      If provided, merges Train + Test for Student training.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(config.SEED)

    # --- 1. Load Data ---

    # Train Data (Fold 0)
    train_images, train_labels, _ = process_and_cache_data(
        config.TRAIN_CSV,
        config.CACHE_TRAIN_IMAGES,
        config.CACHE_TRAIN_LABELS,
        load_cached_data=load_cached_data,
    )

    # Val Data (Fold 0)
    val_images, val_labels, _ = process_and_cache_data(
        config.VAL_CSV,
        config.CACHE_VAL_IMAGES,
        config.CACHE_VAL_LABELS,
        load_cached_data=load_cached_data,
    )

    # Test Data (Fold 1)
    test_images, _, test_ids = process_and_cache_data(
        config.TEST_CSV,
        config.CACHE_TEST_IMAGES,
        "dummy_test_labels.npy",  # Dummy path, not used for loading labels
        config.CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
    )

    # --- 2. Handle Pseudo-Labeling (Student Mode) ---

    if pseudo_labels is not None:
        print("Activating Student Mode: Merging Train and Pseudo-Labeled Test data.")

        # If pseudo_labels is a dict {id: label}, map it to array
        if isinstance(pseudo_labels, dict):
            p_labels_arr = np.zeros(
                (len(test_images), config.NUM_CLASSES), dtype=np.float32
            )
            for i, rid in enumerate(test_ids):
                if rid in pseudo_labels:
                    p_labels_arr[i] = pseudo_labels[rid]
            pseudo_labels = p_labels_arr

        # Ensure pseudo_labels match test_images count
        if len(pseudo_labels) != len(test_images):
            raise ValueError(
                f"Pseudo-labels length {len(pseudo_labels)} mismatch with Test set {len(test_images)}"
            )

        # Concatenate Train (Hard Labels) + Test (Soft Labels)
        final_train_images = np.concatenate([train_images, test_images], axis=0)
        final_train_labels = np.concatenate([train_labels, pseudo_labels], axis=0)

        print(f"Combined Training Set Size: {len(final_train_images)}")
    else:
        final_train_images = train_images
        final_train_labels = train_labels

    # --- 3. Create Datasets ---

    # Debugging Subset
    if config.DEBUG:
        subset_size = min(config.DEBUG_SUBSET_SIZE, len(final_train_images))
        final_train_images = final_train_images[:subset_size]
        final_train_labels = final_train_labels[:subset_size]
        val_images = val_images[:subset_size]
        val_labels = val_labels[:subset_size]
        test_images = test_images[:subset_size]
        test_ids = test_ids[:subset_size]
        print(f"DEBUG MODE: Reduced training size to {subset_size}")

    train_dataset = BirdDataset(
        final_train_images, final_train_labels, transform=get_transforms("train")
    )

    val_dataset = BirdDataset(val_images, val_labels, transform=get_transforms("val"))

    test_dataset = BirdDataset(
        test_images, ids=test_ids, transform=get_transforms("test")
    )

    # --- 4. Create Loaders ---

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
