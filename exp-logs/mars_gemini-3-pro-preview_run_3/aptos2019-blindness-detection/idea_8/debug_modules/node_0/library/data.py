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
seed_everything(Config.seed)


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image to extract the Region of Interest (ROI).
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark or empty
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def pad_to_square(img, value=0):
    """
    Pads a rectangular image to a square shape with black pixels (value=0).
    Preserves aspect ratio without geometric distortion.
    """
    h, w = img.shape[:2]
    if h == w:
        return img

    diff = abs(h - w)
    pad_1 = diff // 2
    pad_2 = diff - pad_1

    if h > w:
        # Pad width (left, right)
        pad_width = ((0, 0), (pad_1, pad_2), (0, 0))
    else:
        # Pad height (top, bottom)
        pad_width = ((pad_1, pad_2), (0, 0), (0, 0))

    img = np.pad(img, pad_width, mode="constant", constant_values=value)
    return img


def preprocess_image(file_path, target_size):
    """
    Reads, crops, pads, and resizes an image.
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            # Create a black image if load fails
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. ROI Extraction (Crop)
        img = crop_image_from_gray(img)

        # 2. Pad to Square (Preserve Geometry)
        img = pad_to_square(img)

        # 3. Resize to Target Resolution
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)


def load_and_process_data(df, image_size, dataset_name, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch and saves to cache.
    Returns numpy arrays of images and labels.
    """
    # Define cache paths
    cache_dir = Config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    images_cache_path = os.path.join(
        cache_dir, f"{dataset_name}_images_{image_size}.npy"
    )
    labels_cache_path = os.path.join(
        cache_dir, f"{dataset_name}_labels_{image_size}.npy"
    )

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(images_cache_path):
        print(
            f"Loading cached {dataset_name} data (size {image_size}) from {cache_dir}..."
        )
        images = np.load(images_cache_path)
        if os.path.exists(labels_cache_path):
            labels = np.load(labels_cache_path)
        else:
            labels = (
                None  # Test set might not have labels in this context if not provided
            )
        return images, labels

    # 2. Process from scratch
    print(f"Processing {dataset_name} data (size {image_size}) from scratch...")

    # Pre-allocate array for efficiency
    n_samples = len(df)
    images = np.zeros((n_samples, image_size, image_size, 3), dtype=np.uint8)

    # Check if diagnosis column exists
    if "diagnosis" in df.columns:
        labels = df["diagnosis"].values.astype(np.float32)
    else:
        labels = None

    # Construct full paths if not already absolute
    # Metadata contains relative paths e.g., "train_images/xxxx.png"
    # Config.input_dir is "./input"
    file_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.input_dir, x)).tolist()
    )

    for i, path in enumerate(file_paths):
        images[i] = preprocess_image(path, image_size)
        if i % 500 == 0:
            print(f"  Processed {i}/{n_samples} images")

    # 3. Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(images_cache_path, images)
    if labels is not None:
        np.save(labels_cache_path, labels)

    return images, labels


class RetinopathyDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image, torch.tensor(0.0, dtype=torch.float32)  # Dummy label for test


def get_transforms(split="train", image_size=512):
    """
    Returns Albumentations transforms.
    Strictly geometric augmentations for training.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose(
            [
                # Geometric Augmentations only
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(image_size, batch_size, load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Validation, and Test sets.

    Args:
        image_size (int): Target resolution (e.g., 512 or 1024).
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    # Debug mode: subset data
    if Config.debug:
        print("DEBUG MODE: Using small subset of data.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(20)

    # Process/Load Data (Caching Logic)
    train_images, train_labels = load_and_process_data(
        df_train, image_size, "train", load_cached_data=load_cached_data
    )
    val_images, val_labels = load_and_process_data(
        df_val, image_size, "val", load_cached_data=load_cached_data
    )
    test_images, _ = load_and_process_data(
        df_test, image_size, "test", load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_images, train_labels, transform=get_transforms("train", image_size)
    )
    val_dataset = RetinopathyDataset(
        val_images, val_labels, transform=get_transforms("val", image_size)
    )
    test_dataset = RetinopathyDataset(
        test_images, None, transform=get_transforms("test", image_size)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
