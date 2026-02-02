import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skmultilearn.model_selection import IterativeStratification
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
    Loads data, combining train and val into a single dev set, and loading test separately.
    """
    cache_files = {
        "dev_imgs": os.path.join(config.CACHE_DIR, "dev_images.npy"),
        "dev_lbls": os.path.join(config.CACHE_DIR, "dev_labels.npy"),
        "test_imgs": os.path.join(config.CACHE_DIR, "test_images.npy"),
        "test_lbls": os.path.join(config.CACHE_DIR, "test_labels.npy"),
    }

    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading data from cache...")
        dev_imgs = np.load(cache_files["dev_imgs"])
        dev_lbls = np.load(cache_files["dev_lbls"])
        test_imgs = np.load(cache_files["test_imgs"])
        test_lbls = np.load(cache_files["test_lbls"])
        return (dev_imgs, dev_lbls), (test_imgs, test_lbls)

    print("Processing data from scratch...")
    # Load and combine metadata
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(config.TEST_METADATA)

    if config.DEBUG:
        dev_df = dev_df.head(64)
        test_df = test_df.head(16)

    print("Processing Development Set (Train + Val)...")
    dev_imgs, dev_lbls = _load_and_process_split(dev_df, config, config.INPUT_ROOT)

    print("Processing Test Set...")
    test_imgs, test_lbls = _load_and_process_split(test_df, config, config.INPUT_ROOT)

    print(f"Saving processed data to cache at {config.CACHE_DIR}...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(cache_files["dev_imgs"], dev_imgs)
    np.save(cache_files["dev_lbls"], dev_lbls)
    np.save(cache_files["test_imgs"], test_imgs)
    np.save(cache_files["test_lbls"], test_lbls)

    return (dev_imgs, dev_lbls), (test_imgs, test_lbls)


def get_folds(X, y, config: Config):
    """
    Generates or loads fold indices using Iterative Stratification.
    """
    fold_path = os.path.join(config.CACHE_DIR, "folds.npy")
    if os.path.exists(fold_path):
        return np.load(fold_path, allow_pickle=True)

    print("Generating folds with Iterative Stratification...")
    stratifier = IterativeStratification(n_splits=config.NUM_FOLDS, order=1)

    folds = []
    for train_idx, val_idx in stratifier.split(X, y):
        folds.append((train_idx, val_idx))

    folds = np.array(folds, dtype=object)
    np.save(fold_path, folds)
    return folds


def get_dataloaders(config: Config, fold: int = 0, load_cached_data: bool = True):
    """
    Creates DataLoaders for a specific fold of K-Fold CV.
    """
    # NOTE: Removed seed_everything here to allow diversity across runs if needed,
    # and to prevent resetting global state inside loops.
    # Cite solution_lesson_node_00063

    # Load Data
    (dev_X, dev_y), (test_X, test_y) = get_data(config, load_cached_data)

    # Get Folds
    folds = get_folds(dev_X, dev_y, config)
    train_idx, val_idx = folds[fold]

    # Subset
    train_X_fold, train_y_fold = dev_X[train_idx], dev_y[train_idx]
    val_X_fold, val_y_fold = dev_X[val_idx], dev_y[val_idx]

    # Transforms
    train_transform = get_transforms(mode="train", img_size=config.IMG_SIZE)
    val_transform = get_transforms(mode="val", img_size=config.IMG_SIZE)

    # Datasets
    train_dataset = BirdDataset(train_X_fold, train_y_fold, transform=train_transform)
    val_dataset = BirdDataset(val_X_fold, val_y_fold, transform=val_transform)
    test_dataset = BirdDataset(test_X, test_y, transform=val_transform)

    # Loaders
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

    return train_loader, val_loader, test_loader, val_idx
