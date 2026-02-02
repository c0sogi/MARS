import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything


def load_dataset_to_ram(metadata_path, subset_name, load_cached_data=True):
    """
    Loads images and metadata into RAM. Uses caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        subset_name (str): Name of the subset (train, val, test) for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, log_filesizes, ids)
            - images: np.array of shape (N, 32, 32, 3), float32, range [0, 1]
            - labels: np.array of shape (N,), float32
            - log_filesizes: np.array of shape (N,), float32 (log1p of bytes)
            - ids: np.array of shape (N,), string (filenames)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_imgs_path = os.path.join(cache_dir, f"cache_{subset_name}_imgs.npy")
    cache_labels_path = os.path.join(cache_dir, f"cache_{subset_name}_labels.npy")
    cache_fsizes_path = os.path.join(cache_dir, f"cache_{subset_name}_fsizes.npy")
    cache_ids_path = os.path.join(cache_dir, f"cache_{subset_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_imgs_path)
            and os.path.exists(cache_labels_path)
            and os.path.exists(cache_fsizes_path)
            and os.path.exists(cache_ids_path)
        ):

            print(f"Loading {subset_name} data from cache...")
            images = np.load(cache_imgs_path)
            labels = np.load(cache_labels_path)
            log_filesizes = np.load(cache_fsizes_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return images, labels, log_filesizes, ids

    print(f"Processing {subset_name} data from scratch...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    fsize_list = []
    id_list = []

    # Pre-construct full paths
    # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
    full_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).tolist()
    )
    ids = df["id"].tolist()
    targets = df["has_cactus"].tolist()

    for i, path in enumerate(full_paths):
        if not os.path.exists(path):
            continue

        # Read Image
        img = cv2.imread(path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 float32
        img = img.astype(np.float32) / 255.0

        # Get file size
        fsize = os.path.getsize(path)

        img_list.append(img)
        label_list.append(targets[i])
        fsize_list.append(np.log1p(fsize))  # Log transform
        id_list.append(ids[i])

    images = np.array(img_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    log_filesizes = np.array(fsize_list, dtype=np.float32)
    ids = np.array(id_list)

    # Save to cache
    np.save(cache_imgs_path, images)
    np.save(cache_labels_path, labels)
    np.save(cache_fsizes_path, log_filesizes)
    np.save(cache_ids_path, ids)

    return images, labels, log_filesizes, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, qualities, transform=None):
        """
        Args:
            images: np.array (N, H, W, C)
            labels: np.array (N,)
            qualities: np.array (N,) - normalized log file sizes
            transform: torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.qualities = qualities
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is already float32 [0, 1]
        img = self.images[idx]
        label = self.labels[idx]
        quality = self.qualities[idx]

        # Apply transforms
        # ToTensor converts HWC numpy to CHW tensor
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "quality": torch.tensor(quality, dtype=torch.float32),
        }


def get_transforms(split="train"):
    """
    Returns the appropriate transforms for training or validation/testing.
    """
    mean = Config.MEAN
    std = Config.STD

    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # HWC numpy -> CHW tensor
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )


def mixup_data(x, y_class, y_quality, alpha=0.2, device=Config.DEVICE):
    """
    Applies Mixup to inputs and targets (both class and quality).
    Returns:
        mixed_x: Mixed images
        y_class_a, y_class_b: Class labels for mixing
        y_quality_a, y_quality_b: Quality labels for mixing
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # We return the pairs and lambda so the loss function can handle the mixing
    y_class_a, y_class_b = y_class, y_class[index]
    y_quality_a, y_quality_b = y_quality, y_quality[index]

    return mixed_x, y_class_a, y_class_b, y_quality_a, y_quality_b, lam


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Orchestrates loading data, normalizing quality targets, and creating DataLoaders.
    """
    # Load raw data (images are 0-1 float, qualities are log1p bytes)
    train_imgs, train_lbls, train_qual, _ = load_dataset_to_ram(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_qual, _ = load_dataset_to_ram(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_qual, test_ids = load_dataset_to_ram(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Debug mode: slice data
    if debug:
        subset_size = 100
        train_imgs, train_lbls, train_qual = (
            train_imgs[:subset_size],
            train_lbls[:subset_size],
            train_qual[:subset_size],
        )
        val_imgs, val_lbls, val_qual = (
            val_imgs[:subset_size],
            val_lbls[:subset_size],
            val_qual[:subset_size],
        )
        test_imgs, test_lbls, test_qual, test_ids = (
            test_imgs[:subset_size],
            test_lbls[:subset_size],
            test_qual[:subset_size],
            test_ids[:subset_size],
        )

    # Normalize Quality Targets (0-1 based on training set statistics)
    q_min = train_qual.min()
    q_max = train_qual.max()

    # Avoid division by zero
    if q_max == q_min:
        q_denom = 1.0
    else:
        q_denom = q_max - q_min

    train_qual_norm = (train_qual - q_min) / q_denom
    val_qual_norm = (val_qual - q_min) / q_denom
    test_qual_norm = (test_qual - q_min) / q_denom

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_qual_norm, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_qual_norm, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_qual_norm, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids, (q_min, q_max)
