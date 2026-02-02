import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config
from library.utils import seed_everything


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Applies Mixup augmentation to a batch of data.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Use the device of the input tensor to ensure compatibility
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BirdDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray): Array of labels (N, num_classes) or None.
            transform: PyTorch transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as uint8 numpy arrays [H, W, 3]
        img_arr = self.images[idx]

        # Convert to PIL Image for torchvision transforms
        img = Image.fromarray(img_arr)

        if self.transform:
            img = self.transform(img)

        # Handle labels
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        else:
            # Return dummy zero labels for test set consistency
            # Assuming 19 classes as per config
            return img, torch.zeros(19, dtype=torch.float32)


def get_transforms(mode="train", img_size=224):
    """
    Returns the transforms for data augmentation and normalization.
    """
    # Standard ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Resize is technically redundant if cached data is already resized,
                # but kept for safety and flexibility.
                transforms.Resize((img_size, img_size)),
                # Horizontal Translation (Time-shifting) via Zero-Padding
                # translate=(0.2, 0) shifts horizontally by up to 20% of width.
                # fill=0 ensures zero-padding.
                transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0),
                # Photometric Augmentation: Brightness and Contrast Jitter
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # Explicitly NO Horizontal Flip as per strategy
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation / Test transforms
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def _load_and_process_split(df, config, input_root):
    """
    Helper to process a dataframe of file paths into numpy arrays.
    Loads BMP spectrograms, converts to RGB (3-channel), resizes, and processes labels.
    """
    images = []
    labels = []

    num_classes = config.NUM_CLASSES

    for _, row in df.iterrows():
        # 1. Image Processing
        # Metadata file_path points to the wav file: essential_data/src_wavs/filename.wav
        # We need to map this to the spectrogram: supplemental_data/spectrograms/filename.bmp
        wav_rel_path = row["file_path"]
        filename = os.path.basename(wav_rel_path)
        bmp_filename = filename.replace(".wav", ".bmp")

        # Construct full path to spectrogram
        img_path = os.path.join(config.SPECTROGRAM_DIR, bmp_filename)

        try:
            with Image.open(img_path) as img:
                # Convert to RGB (3 channels) - this replicates channels if grayscale
                img = img.convert("RGB")
                # Resize to target size using high-quality resampling
                img = img.resize(
                    (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
                )
                images.append(np.array(img))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Fallback: Black image
            images.append(
                np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
            )

        # 2. Label Processing
        label_vec = np.zeros(num_classes, dtype=np.float32)
        lbl_str = str(row["labels"])

        # Check for valid labels (ignore '?' and 'nan')
        if lbl_str != "?" and lbl_str.lower() != "nan":
            try:
                indices = [int(x) for x in lbl_str.split()]
                # Multi-hot encoding
                label_vec[indices] = 1.0
            except ValueError:
                pass

        labels.append(label_vec)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.float32)


def get_data(config: Config, load_cached_data: bool = True):
    """
    Loads data, processing it from scratch or loading from cache.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: ((train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls))
    """
    # Define cache file paths
    cache_files = {
        "train_imgs": os.path.join(config.CACHE_DIR, "train_images.npy"),
        "train_lbls": os.path.join(config.CACHE_DIR, "train_labels.npy"),
        "val_imgs": os.path.join(config.CACHE_DIR, "val_images.npy"),
        "val_lbls": os.path.join(config.CACHE_DIR, "val_labels.npy"),
        "test_imgs": os.path.join(config.CACHE_DIR, "test_images.npy"),
        "test_lbls": os.path.join(config.CACHE_DIR, "test_labels.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_imgs = np.load(cache_files["train_imgs"])
        train_lbls = np.load(cache_files["train_lbls"])
        val_imgs = np.load(cache_files["val_imgs"])
        val_lbls = np.load(cache_files["val_lbls"])
        test_imgs = np.load(cache_files["test_imgs"])
        test_lbls = np.load(cache_files["test_lbls"])
        return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)

    print("Processing data from scratch...")

    # Load Metadata CSVs
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)

    # Debug mode: use a small subset
    if config.DEBUG:
        print("DEBUG Mode: Using subset of data.")
        train_df = train_df.head(32)
        val_df = val_df.head(16)
        test_df = test_df.head(16)

    # Process each split
    print("Processing Training Set...")
    train_imgs, train_lbls = _load_and_process_split(
        train_df, config, config.INPUT_ROOT
    )

    print("Processing Validation Set...")
    val_imgs, val_lbls = _load_and_process_split(val_df, config, config.INPUT_ROOT)

    print("Processing Test Set...")
    test_imgs, test_lbls = _load_and_process_split(test_df, config, config.INPUT_ROOT)

    # Save to cache
    print(f"Saving processed data to cache at {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(cache_files["train_imgs"], train_imgs)
    np.save(cache_files["train_lbls"], train_lbls)
    np.save(cache_files["val_imgs"], val_imgs)
    np.save(cache_files["val_lbls"], val_lbls)
    np.save(cache_files["test_imgs"], test_imgs)
    np.save(cache_files["test_lbls"], test_lbls)

    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)


def get_dataloaders(config: Config, load_cached_data: bool = True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(config.SEED)

    # Load Data
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_data(
        config, load_cached_data
    )

    # Define Transforms
    train_transform = get_transforms(mode="train", img_size=config.IMG_SIZE)
    val_transform = get_transforms(mode="val", img_size=config.IMG_SIZE)

    # Create Datasets
    train_dataset = BirdDataset(train_X, train_y, transform=train_transform)
    val_dataset = BirdDataset(val_X, val_y, transform=val_transform)
    test_dataset = BirdDataset(test_X, test_y, transform=val_transform)

    # Create DataLoaders
    # drop_last=True for training to ensure stable Batch Norm statistics
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
