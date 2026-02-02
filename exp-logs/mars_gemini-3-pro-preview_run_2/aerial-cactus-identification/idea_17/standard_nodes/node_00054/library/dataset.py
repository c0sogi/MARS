import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


def get_transforms(stage="train"):
    """
    Returns the torchvision transforms for the specified stage.

    Args:
        stage (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transforms.
    """
    transform_list = [
        transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0, 1.0]
    ]

    if stage == "train":
        # Light augmentation: only flips allowed
        transform_list.append(transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB))
        transform_list.append(transforms.RandomVerticalFlip(p=Config.AUG_VFLIP_PROB))

    return transforms.Compose(transform_list)


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N,).
            ids (np.ndarray, optional): Array of IDs.
            transform (callable, optional): Transform to apply.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        img = self.images[idx]

        # Apply transforms (converts to Tensor CHW)
        if self.transform:
            img = self.transform(img)

        item = {"image": img}

        if self.labels is not None:
            # Float tensor for BCEWithLogitsLoss
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata and images, with caching to .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Adjust prefix for debug mode to avoid cache collisions
    if Config.DEBUG:
        cache_prefix = f"{cache_prefix}_debug"

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(id_cache_path):
            try:
                images = np.load(img_cache_path)
                ids = np.load(id_cache_path, allow_pickle=True)

                labels = None
                if os.path.exists(lbl_cache_path):
                    labels = np.load(lbl_cache_path)

                return images, labels, ids
            except Exception:
                # If loading fails, fall through to recompute
                pass

    # 2. Compute from scratch
    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    image_list = []
    label_list = []
    id_list = []

    # Pre-allocate if possible or just append (dataset is small enough for append)
    for _, row in df.iterrows():
        # Construct full path: input_dir + relative_path_from_metadata
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_list.append(img)
        id_list.append(row["id"])

        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    images = np.array(image_list, dtype=np.uint8)
    ids = np.array(id_list)

    if label_list:
        labels = np.array(label_list, dtype=np.float32)
    else:
        labels = None

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(id_cache_path, ids)
    if labels is not None:
        np.save(lbl_cache_path, labels)

    return images, labels, ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """

    # Load Data
    train_imgs, train_lbls, train_ids = load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # Create DataLoaders
    # Pin memory speeds up host to device copy
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
