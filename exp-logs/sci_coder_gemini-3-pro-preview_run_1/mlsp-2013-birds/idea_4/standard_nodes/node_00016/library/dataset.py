import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything


def spec_augment(
    image,
    num_mask=2,
    freq_masking_max_percentage=0.10,
    time_masking_max_percentage=0.15,
):
    """
    Applies SpecAugment (Time and Frequency Masking) to an image.

    Args:
        image (np.ndarray): Input image of shape (H, W, C).
        num_mask (int): Number of masks to apply.
        freq_masking_max_percentage (float): Max percentage of height to mask.
        time_masking_max_percentage (float): Max percentage of width to mask.

    Returns:
        np.ndarray: Augmented image.
    """
    h, w, c = image.shape
    aug_image = image.copy()

    for _ in range(num_mask):
        # Frequency masking (Horizontal strips -> Axis 0 is Height/Freq)
        f = int(np.random.uniform(0, freq_masking_max_percentage) * h)
        f0 = np.random.randint(0, max(1, h - f))
        aug_image[f0 : f0 + f, :, :] = 0.0

        # Time masking (Vertical strips -> Axis 1 is Width/Time)
        t = int(np.random.uniform(0, time_masking_max_percentage) * w)
        t0 = np.random.randint(0, max(1, w - t))
        aug_image[:, t0 : t0 + t, :] = 0.0

    return aug_image


class BirdDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N, NumClasses).
            transform (A.Compose): Albumentations transforms.
            augment (bool): Whether to apply SpecAugment.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)

        # Apply SpecAugment if training
        if self.augment:
            image = spec_augment(image)

        # Apply Albumentations (Normalization + ToTensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = torch.tensor(image).permute(2, 0, 1)

        return image, label


def get_transforms(phase):
    """
    Returns the Albumentations transformation pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def process_data(df, load_cached_data=True, cache_prefix="train"):
    """
    Loads images listed in the dataframe, processes them (resize, normalize),
    and handles caching to .npy files.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_prefix (str): Prefix for cache filenames (e.g., 'train', 'val').

    Returns:
        tuple: (X, y) numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    x_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_images.npy")
    y_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading {cache_prefix} data from cache...")
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            return X, y
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {cache_prefix} data from scratch...")
    X = []
    y = []

    # Identify label columns
    # Explicitly select columns based on Config.NUM_CLASSES (Cite debug_lesson_1)
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    for idx, row in df.iterrows():
        # Construct path to spectrogram
        # Metadata file_path example: "essential_data/src_wavs/PC10_....wav"
        # Spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension
        wav_rel_path = row["file_path"]
        wav_filename = os.path.basename(wav_rel_path)
        bmp_filename = os.path.splitext(wav_filename)[0] + ".bmp"

        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        if os.path.exists(img_path):
            # Load as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Handle corrupt image
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
            else:
                # Resize: cv2.resize takes (width, height)
                img = cv2.resize(
                    img,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_CUBIC,
                )

            # Stack to 3 channels (H, W, 3)
            img = np.stack([img, img, img], axis=-1)

            # Normalize to 0-1
            img = img.astype(np.float32) / 255.0

            X.append(img)
            y.append(row[label_cols].values.astype(np.float32))
        else:
            # Handle missing file (though EDA showed none)
            print(f"Warning: File not found {img_path}")
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.float32)
            X.append(img)
            y.append(row[label_cols].values.astype(np.float32))

    X = np.array(X)
    y = np.array(y)

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(x_path, X)
    np.save(y_path, y)

    return X, y


def get_datasets(load_cached_data=True):
    """
    Main function to prepare datasets for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    seed_everything(Config.SEED)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Subset data
    if Config.DEBUG:
        print(f"DEBUG MODE: Using subset of {Config.DEBUG_SUBSET_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # Process Data (Load Images -> Resize -> Cache)
    X_train, y_train = process_data(df_train, load_cached_data, "train")
    X_val, y_val = process_data(df_val, load_cached_data, "val")
    X_test, y_test = process_data(df_test, load_cached_data, "test")

    # Create Datasets
    # Train: With Augmentation
    train_dataset = BirdDataset(
        X_train, y_train, transform=get_transforms("train"), augment=True
    )

    # Val: No Augmentation
    val_dataset = BirdDataset(
        X_val, y_val, transform=get_transforms("val"), augment=False
    )

    # Test: No Augmentation
    test_dataset = BirdDataset(
        X_test, y_test, transform=get_transforms("test"), augment=False
    )

    return train_dataset, val_dataset, test_dataset
