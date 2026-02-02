import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
CACHE_DIR = "./working/idea_21"
NUM_CLASSES = 19
IMG_SIZE = 224


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles images loaded from numpy arrays (cached) and applies transforms.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of multi-hot labels (N, NumClasses).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as uint8 (H, W, C)
        img_arr = self.images[idx]

        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(img_arr)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(mode="train"):
    """
    Returns the transformation pipeline for training or validation/testing.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Resize is technically redundant if cached data is already resized,
                # but ensures safety if input changes.
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                # Horizontal Shift (Zero-padding)
                # translate=(0.2, 0) means shift horizontally by up to 20% of width
                # fill=0 ensures zero-padding, preserving temporal causality
                transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0),
                # Photometric Augmentation (Brightness and Contrast Jitter)
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Validation / Test / TTA base
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def process_and_cache_data(split, load_cached_data=True):
    """
    Loads data from metadata CSVs and spectrogram files, processes them (resize, 3-channel),
    and caches them as numpy arrays.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{split}_images.npy")
    lbl_cache_path = os.path.join(CACHE_DIR, f"{split}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(img_cache_path):
        # Check if labels exist for non-test splits
        if split == "test" or os.path.exists(lbl_cache_path):
            print(f"Loading cached {split} data from {CACHE_DIR}...")
            images = np.load(img_cache_path)
            if split != "test":
                labels = np.load(lbl_cache_path)
            else:
                labels = None
            return images, labels

    # 2. Process from scratch
    print(f"Processing {split} data from scratch...")

    csv_path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    images_list = []
    labels_list = []

    # Use appropriate resampling filter based on PIL version
    resample_method = getattr(Image, "Resampling", Image).BILINEAR

    for idx, row in df.iterrows():
        # Construct path to spectrogram
        # Original: essential_data/src_wavs/filename.wav
        # Target: supplemental_data/spectrograms/filename.bmp
        orig_filename = os.path.basename(row["file_path"])
        bmp_filename = orig_filename.replace(".wav", ".bmp")
        bmp_path = os.path.join(SPECTROGRAM_DIR, bmp_filename)

        # Load and Process Image
        try:
            with Image.open(bmp_path) as img:
                # Replicate channels to create Pseudo-RGB (3 channels)
                img = img.convert("RGB")

                # Resize to 224x224
                img = img.resize((IMG_SIZE, IMG_SIZE), resample=resample_method)

                images_list.append(np.array(img))
        except Exception as e:
            print(f"Warning: Could not load {bmp_path}. Using black image. Error: {e}")
            images_list.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        # Process Labels (if not test)
        if split != "test":
            label_vec = np.zeros(NUM_CLASSES, dtype=int)
            label_str = str(row["labels"])
            # Check for valid label string
            if label_str and label_str != "?" and label_str.lower() != "nan":
                try:
                    indices = [int(x) for x in label_str.split()]
                    label_vec[indices] = 1
                except ValueError:
                    pass  # Keep zero vector if parsing fails
            labels_list.append(label_vec)

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)

    # Save images to cache
    np.save(img_cache_path, images)

    if split != "test":
        labels = np.array(labels_list, dtype=np.float32)
        np.save(lbl_cache_path, labels)
    else:
        labels = None

    return images, labels


def get_datasets(load_cached_data=True):
    """
    Factory function to get Train, Validation, and Test datasets.
    Ensures reproducibility by seeding everything.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    seed_everything(42)

    # Train
    train_imgs, train_lbls = process_and_cache_data("train", load_cached_data)
    train_dataset = BirdDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    # Validation
    val_imgs, val_lbls = process_and_cache_data("val", load_cached_data)
    val_dataset = BirdDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    # Test
    test_imgs, _ = process_and_cache_data("test", load_cached_data)
    test_dataset = BirdDataset(
        test_imgs,
        None,
        transform=get_transforms("val"),  # Use val transforms (resize/norm) for test
    )

    return train_dataset, val_dataset, test_dataset
