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
from library.utils import seed_everything


def get_transforms(phase="train"):
    """
    Constructs the augmentation pipeline using Albumentations.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform composition.
    """
    height, width = Config.IMG_SIZE

    if phase == "train":
        transforms = [
            # Ensure size is correct (though caching handles this, redundancy is safe)
            A.Resize(height=height, width=width),
            # Zero-Padding Time Shift (Horizontal Translation)
            # We shift only on X axis (time), with zero padding (BORDER_CONSTANT)
            # This preserves temporal causality unlike wrapping.
            A.ShiftScaleRotate(
                shift_limit_x=0.1 if Config.ENABLE_TIME_SHIFT else 0.0,
                shift_limit_y=0.0,
                scale_limit=0.0,
                rotate_limit=0.0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            # Photometric Augmentations to mimic recording variations (Gain/SNR)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5 if Config.ENABLE_PHOTOMETRIC else 0.0,
            ),
            # Normalization (ImageNet stats)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        # Validation/Test: Resize and Normalize only
        transforms = [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


def load_image(file_path):
    """
    Loads a spectrogram image, converts to RGB, and resizes.

    Args:
        file_path (str): Relative path from metadata (pointing to .wav).

    Returns:
        np.ndarray: The processed image (H, W, 3).
    """
    # Convert .wav path in essential_data to .bmp path in supplemental_data
    # Example: essential_data/src_wavs/PC10...wav -> PC10...bmp
    filename = os.path.basename(file_path)
    filename = filename.replace(".wav", ".bmp")
    full_path = os.path.join(Config.SPECTROGRAM_DIR, filename)

    # Load image
    image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    # Handle missing files or errors by returning a black image
    if image is None:
        return np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8)

    # Apply 3-Channel Rule: Replicate single channel to RGB
    # This ensures compatibility with pre-trained models (ResNet/EfficientNet)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # (H, W, 1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to target size immediately to save cache space and compute during training
    # cv2.resize uses (width, height)
    image = cv2.resize(image, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))

    return image


def parse_labels(label_str):
    """
    Parses a label string into a multi-hot binary vector.

    Args:
        label_str (str): Space-separated indices (e.g., "0 4").

    Returns:
        np.ndarray: Binary vector of shape (NUM_CLASSES,).
    """
    vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
    if pd.isna(label_str) or label_str == "?" or str(label_str).strip() == "":
        return vec

    try:
        indices = [int(x) for x in str(label_str).split()]
        for idx in indices:
            if 0 <= idx < Config.NUM_CLASSES:
                vec[idx] = 1.0
    except ValueError:
        pass
    return vec


def get_data_arrays(df, cache_prefix, load_cached_data=True):
    """
    Loads images and labels for a dataframe, utilizing caching to speed up subsequent runs.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_prefix (str): Prefix for cache filenames.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels) numpy arrays.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    images_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    labels_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(images_path) and os.path.exists(labels_path):
        try:
            images = np.load(images_path)
            labels = np.load(labels_path)
            # Verify consistency
            if len(images) == len(df):
                return images, labels
        except Exception:
            pass  # Fallback to processing from scratch

    # Process data from scratch
    images = []
    labels = []

    for _, row in df.iterrows():
        img = load_image(row["file_path"])
        lbl = parse_labels(row["labels"])
        images.append(img)
        labels.append(lbl)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)

    # Save to cache
    np.save(images_path, images)
    np.save(labels_path, labels)

    return images, labels


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Wraps pre-loaded numpy arrays for efficiency.
    """

    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, label


def get_dataloaders(fold=0, load_cached_data=True):
    """
    Creates Training and Validation DataLoaders for a specific fold.
    Uses Iterative Stratification on the combined Train+Val metadata to ensure
    balanced multi-label distribution across folds.

    Args:
        fold (int): Fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data arrays.

    Returns:
        tuple: (train_loader, val_loader)
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine to form the full development set for Cross-Validation
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Data Arrays (with caching)
    images, labels = get_data_arrays(
        full_df, "full_dev", load_cached_data=load_cached_data
    )

    # Handle Debugging: Subset data if enabled
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(images))
        images = images[:subset_size]
        labels = labels[:subset_size]

    # Perform Iterative Stratification
    # We use dummy X because stratification is based on y (labels)
    X_dummy = np.zeros((len(labels), 1))
    stratifier = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)

    # split() returns a generator of indices
    # Note: IterativeStratification is deterministic given fixed input and seed
    splits = list(stratifier.split(X_dummy, labels))
    train_indices, val_indices = splits[fold]

    # Split data based on indices
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]

    # Create Datasets
    train_dataset = BirdDataset(
        train_images, train_labels, transforms=get_transforms("train")
    )

    val_dataset = BirdDataset(val_images, val_labels, transforms=get_transforms("val"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Norm stability and Mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates a DataLoader for the Test set.

    Args:
        load_cached_data (bool): Whether to use cached data arrays.

    Returns:
        tuple: (test_loader, test_ids)
    """
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    images, labels = get_data_arrays(
        test_meta, "test", load_cached_data=load_cached_data
    )

    # Handle Debugging
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(images))
        images = images[:subset_size]
        labels = labels[:subset_size]
        test_ids = test_meta["rec_id"].values[:subset_size]
    else:
        test_ids = test_meta["rec_id"].values

    dataset = BirdDataset(
        images,
        labels,  # Labels are placeholders/hidden
        transforms=get_transforms("test"),
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, test_ids
