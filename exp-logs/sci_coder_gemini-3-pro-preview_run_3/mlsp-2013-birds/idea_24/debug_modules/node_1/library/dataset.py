import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# Set seeds for reproducibility
seed_everything(42)

# Constants
CACHE_DIR = "./working/idea_24/"
INPUT_DIR = "./input"
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
NUM_CLASSES = 19
IMG_SIZE = 224


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles pseudo-RGB spectrograms and multi-label targets.
    """

    def __init__(self, images, labels, rec_ids, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray): Array of multi-hot labels (N, NumClasses).
            rec_ids (np.ndarray): Array of recording IDs (N,).
            transforms (albumentations.Compose): Transformations to apply.
        """
        self.images = images
        self.labels = labels
        self.rec_ids = rec_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        label = self.labels[idx]
        rec_id = self.rec_ids[idx]

        # Apply augmentations/transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Convert label to float tensor for BCEWithLogitsLoss
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, rec_id


def get_transforms(data_type="train"):
    """
    Returns the augmentation pipeline.

    Args:
        data_type (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_type == "train":
        return A.Compose(
            [
                # Ensure strictly 224x224
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                # Horizontal Translation (Time-shifting) via Zero-Padding
                # shift_limit_x=0.2 -> +/- 20% shift
                # border_mode=cv2.BORDER_CONSTANT with value=0 adds black padding (silence)
                A.ShiftScaleRotate(
                    shift_limit_x=0.2,
                    shift_limit_y=0.0,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentation: Brightness and Contrast Jitter
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization and Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test pipeline: Resize -> Normalize -> Tensor
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def load_data(
    metadata_path,
    load_cached_data=True,
    cache_dir=CACHE_DIR,
    split_name="train",
    max_samples=None,
):
    """
    Loads data from metadata CSV, processes images (read, resize, RGB-replicate),
    and caches the result as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to try loading from cache.
        cache_dir (str): Directory to store cached .npy files.
        split_name (str): Name of the split (e.g., 'train', 'val') for cache naming.
        max_samples (int, optional): Limit number of samples for debugging.

    Returns:
        tuple: (images, labels, rec_ids) as numpy arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_img_path = os.path.join(cache_dir, f"{split_name}_images.npy")
    cache_lbl_path = os.path.join(cache_dir, f"{split_name}_labels.npy")
    cache_id_path = os.path.join(cache_dir, f"{split_name}_ids.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_img_path)
        and os.path.exists(cache_lbl_path)
        and os.path.exists(cache_id_path)
    ):
        # print(f"Loading {split_name} data from cache...")
        images = np.load(cache_img_path)
        labels = np.load(cache_lbl_path)
        rec_ids = np.load(cache_id_path)
    else:
        # 2. Process from scratch
        # print(f"Processing {split_name} data from source...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        img_list = []
        lbl_list = []
        id_list = []

        for _, row in df.iterrows():
            # --- Image Processing ---
            # Metadata points to .wav in essential_data/src_wavs/
            # We need to map this to .bmp in supplemental_data/spectrograms/
            wav_rel_path = row["file_path"]  # e.g., essential_data/src_wavs/PC10_...wav
            filename = os.path.basename(wav_rel_path)
            bmp_filename = os.path.splitext(filename)[0] + ".bmp"
            bmp_path = os.path.join(SPECTROGRAM_DIR, bmp_filename)

            if os.path.exists(bmp_path):
                # Read as grayscale
                img_gray = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    # Fallback for corrupt file (unlikely given verification)
                    img_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    # Resize to 224x224 immediately to save space and ensure consistency
                    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                    # Replicate channels: 1 -> 3 (Pseudo-RGB)
                    img_rgb = cv2.merge([img_resized, img_resized, img_resized])
            else:
                # Fallback for missing file
                img_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            img_list.append(img_rgb)
            id_list.append(row["rec_id"])

            # --- Label Processing ---
            label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            lbl_str = str(row["labels"])

            # Parse space-separated indices (e.g., "0 4")
            if lbl_str != "?" and lbl_str.lower() != "nan" and lbl_str.strip():
                try:
                    indices = [int(x) for x in lbl_str.split()]
                    # Clip indices just in case
                    indices = [x for x in indices if 0 <= x < NUM_CLASSES]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass

            lbl_list.append(label_vec)

        # Convert to numpy arrays
        images = np.array(img_list, dtype=np.uint8)
        labels = np.array(lbl_list, dtype=np.float32)
        rec_ids = np.array(id_list, dtype=np.int64)

        # Save to cache
        np.save(cache_img_path, images)
        np.save(cache_lbl_path, labels)
        np.save(cache_id_path, rec_ids)

    # 3. Handle max_samples (Debugging)
    if max_samples is not None:
        images = images[:max_samples]
        labels = labels[:max_samples]
        rec_ids = rec_ids[:max_samples]

    return images, labels, rec_ids


def get_datasets(load_cached_data=True, max_samples=None):
    """
    Helper function to load all datasets (train, val, test) initialized with
    their respective transforms.

    Args:
        load_cached_data (bool): Whether to use caching.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Train
    train_img, train_lbl, train_ids = load_data(
        "./metadata/train.csv",
        load_cached_data,
        split_name="train",
        max_samples=max_samples,
    )
    train_dataset = BirdDataset(
        train_img, train_lbl, train_ids, transforms=get_transforms("train")
    )

    # Validation
    val_img, val_lbl, val_ids = load_data(
        "./metadata/val.csv",
        load_cached_data,
        split_name="val",
        max_samples=max_samples,
    )
    val_dataset = BirdDataset(
        val_img, val_lbl, val_ids, transforms=get_transforms("val")
    )

    # Test
    test_img, test_lbl, test_ids = load_data(
        "./metadata/test.csv",
        load_cached_data,
        split_name="test",
        max_samples=max_samples,
    )
    test_dataset = BirdDataset(
        test_img, test_lbl, test_ids, transforms=get_transforms("test")
    )

    return train_dataset, val_dataset, test_dataset
