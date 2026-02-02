import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skmultilearn.model_selection import IterativeStratification
from library.config import Config
from library.utils import save_cache, load_cache, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles 3-channel conversion and on-the-fly augmentation.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            labels (np.ndarray, optional): Multi-hot encoded labels (N, Num_Classes).
            ids (np.ndarray, optional): Recording IDs.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (stored as uint8 numpy array)
        img_arr = self.images[idx]

        # Convert to PIL Image
        # Input is (H, W), mode 'L'
        image = Image.fromarray(img_arr, mode="L")

        # Apply transforms
        # Note: Channel replication to 3 channels is handled in the transform pipeline
        if self.transform:
            image = self.transform(image)

        rec_id = self.ids[idx] if self.ids is not None else -1

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label, rec_id
        else:
            return image, rec_id


def get_transforms(mode="train"):
    """
    Returns the transformation pipeline based on the mode.
    Strictly adheres to the strategy:
    - 3-Channel Replication
    - Horizontal Translation (Zero-Padding)
    - Photometric Jitter
    - No Horizontal Flip
    """
    # Normalization constants for ImageNet pre-trained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Replicate 1 channel to 3 channels (Pseudo-RGB)
                transforms.Grayscale(num_output_channels=3),
                # Horizontal Translation (Time-Shifting) using RandomAffine
                # translate=(tx, ty): tx is fraction of width. fill=0 ensures zero-padding.
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.0), fill=0),
                # Photometric Augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # Convert to Tensor
                transforms.ToTensor(),
                # Normalize
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation / Test / TTA base
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def load_image_data(metadata_df, image_dir, image_size, debug_max=None):
    """
    Helper to load images and process labels from a metadata DataFrame.
    """
    images = []
    labels = []
    ids = []

    # Limit samples for debugging if requested
    if debug_max is not None:
        metadata_df = metadata_df.iloc[:debug_max]

    for _, row in metadata_df.iterrows():
        # Construct full path. Metadata file_path is relative to input root.
        # However, Config.SPECTROGRAM_DIR points to supplemental_data/spectrograms.
        # The metadata file_path is like "essential_data/src_wavs/PC10...wav".
        # We need to map this to the spectrogram path.
        # The spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension.

        # Extract filename base
        wav_path = row["file_path"]
        base_name = os.path.basename(wav_path)
        file_id = os.path.splitext(base_name)[0]
        bmp_filename = f"{file_id}.bmp"
        bmp_path = os.path.join(image_dir, bmp_filename)

        try:
            # Load BMP image
            with Image.open(bmp_path) as img:
                # Resize strictly to 224x224
                img = img.resize(
                    (image_size, image_size), resample=Image.Resampling.BILINEAR
                )
                # Convert to grayscale numpy array
                img_arr = np.array(img.convert("L"))
                images.append(img_arr)
        except Exception as e:
            print(f"Warning: Could not load image {bmp_path}: {e}")
            continue

        ids.append(row["rec_id"])

        # Process labels if present
        if "labels" in row and row["labels"] != "?":
            # Create multi-hot vector
            lbl_vec = np.zeros(Config.NUM_CLASSES, dtype=int)
            lbl_str = str(row["labels"]).strip()
            if lbl_str:
                indices = [int(x) for x in lbl_str.split()]
                lbl_vec[indices] = 1
            labels.append(lbl_vec)
        else:
            # Placeholder for test set
            labels.append(np.zeros(Config.NUM_CLASSES, dtype=int))

    return np.array(images), np.array(labels), np.array(ids)


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches the dataset.
    Merges Train and Validation metadata to form a full Development set.
    """
    cache_files = {
        "dev_images": "dev_images.npy",
        "dev_labels": "dev_labels.npy",
        "dev_ids": "dev_ids.npy",
        "test_images": "test_images.npy",
        "test_ids": "test_ids.npy",
    }

    # Try loading from cache
    if load_cached_data:
        dev_images = load_cache(cache_files["dev_images"])
        dev_labels = load_cache(cache_files["dev_labels"])
        dev_ids = load_cache(cache_files["dev_ids"])
        test_images = load_cache(cache_files["test_images"])
        test_ids = load_cache(cache_files["test_ids"])

        if all(
            x is not None
            for x in [dev_images, dev_labels, dev_ids, test_images, test_ids]
        ):
            return (dev_images, dev_labels, dev_ids), (test_images, test_ids)

    # If cache miss or force reload, process from scratch

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Merge Train and Val for full development set
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    # 2. Load Images
    dev_images, dev_labels, dev_ids = load_image_data(
        dev_df, Config.SPECTROGRAM_DIR, Config.IMAGE_SIZE, Config.DEBUG_MAX_SAMPLES
    )

    test_images, _, test_ids = load_image_data(
        test_df, Config.SPECTROGRAM_DIR, Config.IMAGE_SIZE, Config.DEBUG_MAX_SAMPLES
    )

    # 3. Save to Cache
    save_cache(dev_images, cache_files["dev_images"])
    save_cache(dev_labels, cache_files["dev_labels"])
    save_cache(dev_ids, cache_files["dev_ids"])
    save_cache(test_images, cache_files["test_images"])
    save_cache(test_ids, cache_files["test_ids"])

    return (dev_images, dev_labels, dev_ids), (test_images, test_ids)


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Generates DataLoaders for a specific fold using Iterative Stratification.

    Args:
        fold_idx (int): Index of the fold to use as validation (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        train_loader, val_loader
    """
    # Load full development set
    (images, labels, ids), _ = prepare_data(load_cached_data=load_cached_data)

    # Perform Iterative Stratification
    # We use the indices to split
    X_indices = np.zeros((len(labels), 1))  # Dummy X for splitter

    stratifier = IterativeStratification(
        n_splits=Config.NUM_FOLDS, order=1, random_state=Config.SEED
    )

    splits = list(stratifier.split(X_indices, labels))
    train_indices, val_indices = splits[fold_idx]

    # Create subsets
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    train_ids = ids[train_indices]

    val_images = images[val_indices]
    val_labels = labels[val_indices]
    val_ids = ids[val_indices]

    # Create Datasets
    train_dataset = BirdDataset(
        train_images, train_labels, train_ids, transform=get_transforms(mode="train")
    )

    val_dataset = BirdDataset(
        val_images, val_labels, val_ids, transform=get_transforms(mode="valid")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Generates DataLoader for the test set.
    """
    _, (test_images, test_ids) = prepare_data(load_cached_data=load_cached_data)

    test_dataset = BirdDataset(
        test_images, labels=None, ids=test_ids, transform=get_transforms(mode="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader
