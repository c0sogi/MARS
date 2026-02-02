import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_38"


def load_data_from_metadata(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV, reading images and caching them as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        tuple: (images, targets, ids)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    images_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    targets_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_targets.npy")
    ids_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(targets_cache_path)
        and os.path.exists(ids_cache_path)
    ):
        print(f"Loading cached data for {cache_prefix} from {CACHE_DIR}...")
        try:
            images = np.load(images_cache_path)
            targets = np.load(targets_cache_path)
            ids = np.load(ids_cache_path, allow_pickle=True)
            return images, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data...")

    # 2. Process from scratch
    print(f"Processing data for {cache_prefix} from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays for efficiency
    num_samples = len(df)
    images = np.zeros((num_samples, 32, 32, 3), dtype=np.uint8)
    targets = np.zeros(num_samples, dtype=np.float32)
    ids = np.empty(num_samples, dtype=object)

    for idx, row in df.iterrows():
        # Construct full path
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            # In case of read failure, we default to a black image to avoid crashing,
            # though the metadata check implies all files exist.
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[idx] = img
        targets[idx] = row["has_cactus"]
        ids[idx] = row["id"]

    # 3. Save to cache
    np.save(images_cache_path, images)
    np.save(targets_cache_path, targets)
    np.save(ids_cache_path, ids)
    print(f"Cached data for {cache_prefix} saved to {CACHE_DIR}")

    return images, targets, ids


class CactusDataset(Dataset):
    def __init__(self, images, targets=None, ids=None, transform=None, return_id=False):
        """
        Custom Dataset for Cactus images.

        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            targets (np.ndarray, optional): Array of targets (N,).
            ids (np.ndarray, optional): Array of image IDs (N,).
            transform (callable, optional): Transform to be applied on a sample.
            return_id (bool): If True, returns (image, id). Else returns (image, target).
        """
        self.images = images
        self.targets = targets
        self.ids = ids
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C)
        img = self.images[idx]

        # Apply transforms (e.g., ToTensor, Flips)
        if self.transform:
            img = self.transform(img)

        if self.return_id:
            # Return ID for test submission
            return img, self.ids[idx]
        else:
            # Return target for training/validation
            target = self.targets[idx] if self.targets is not None else 0.0
            return img, torch.tensor(target, dtype=torch.float32)


def get_dataloaders(
    batch_size=32, num_workers=2, load_cached_data=True, debug_size=None
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached .npy files.
        debug_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Metadata paths
    train_meta = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Load raw data (or from cache)
    train_imgs, train_targets, train_ids = load_data_from_metadata(
        train_meta, "train", load_cached_data
    )
    val_imgs, val_targets, val_ids = load_data_from_metadata(
        val_meta, "val", load_cached_data
    )
    test_imgs, test_targets, test_ids = load_data_from_metadata(
        test_meta, "test", load_cached_data
    )

    # Apply debug slicing if requested
    if debug_size is not None:
        print(f"Debug mode: Limiting dataset sizes to {debug_size} samples.")
        train_imgs = train_imgs[:debug_size]
        train_targets = train_targets[:debug_size]
        val_imgs = val_imgs[:debug_size]
        val_targets = val_targets[:debug_size]
        test_imgs = test_imgs[:debug_size]
        test_ids = test_ids[:debug_size]

    # Define Transforms
    # Train: ToTensor (scales [0,255]->[0,1]) + Light Augmentation
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # Val/Test: ToTensor only
    eval_transform = transforms.Compose([transforms.ToTensor()])

    # Instantiate Datasets
    train_dataset = CactusDataset(
        train_imgs, train_targets, ids=train_ids, transform=train_transform
    )
    val_dataset = CactusDataset(
        val_imgs, val_targets, ids=val_ids, transform=eval_transform
    )
    # Test dataset returns IDs for submission generation
    test_dataset = CactusDataset(
        test_imgs, targets=None, ids=test_ids, transform=eval_transform, return_id=True
    )

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
