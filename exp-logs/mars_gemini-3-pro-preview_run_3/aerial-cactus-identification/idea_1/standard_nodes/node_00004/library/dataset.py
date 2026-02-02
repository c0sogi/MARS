import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import seed_everything


def load_and_preprocess_data(
    metadata_path, input_dir, cache_name, load_cached_data=True
):
    """
    Loads images based on metadata, converts to arrays, and handles caching.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        input_dir (str): Root directory containing the images.
        cache_name (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        ids (numpy array): Array of image IDs.
        images (numpy array): Array of image data (N, 32, 32, 3) in uint8.
        labels (numpy array): Array of labels.
    """
    cache_dir = "./working/idea_1"
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_ids = os.path.join(cache_dir, f"{cache_name}_ids.npy")
    cache_path_imgs = os.path.join(cache_dir, f"{cache_name}_imgs.npy")
    cache_path_lbls = os.path.join(cache_dir, f"{cache_name}_lbls.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_path_ids)
            and os.path.exists(cache_path_imgs)
            and os.path.exists(cache_path_lbls)
        ):
            print(f"Loading {cache_name} data from cache...")
            ids = np.load(cache_path_ids, allow_pickle=True)
            images = np.load(cache_path_imgs)
            labels = np.load(cache_path_lbls)
            return ids, images, labels

    # 2. Process from scratch
    print(f"Processing {cache_name} data from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    ids = []
    images = []
    labels = []

    for _, row in df.iterrows():
        img_id = row["id"]
        rel_path = row["file_path"]
        # Handle test set where label might be a placeholder or missing
        label = row["has_cactus"] if "has_cactus" in row else 0.0

        full_path = os.path.join(input_dir, rel_path)
        img = cv2.imread(full_path)

        if img is None:
            # Fallback: create a black image to prevent crash, though metadata should be verified
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ids.append(img_id)
        images.append(img)
        labels.append(label)

    ids = np.array(ids)
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)

    # 3. Save to cache
    print(f"Saving {cache_name} data to cache at {cache_dir}...")
    np.save(cache_path_ids, ids)
    np.save(cache_path_imgs, images)
    np.save(cache_path_lbls, labels)

    return ids, images, labels


class CactusDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) in uint8
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Transforms expect PIL Image or Tensor.
            # ToTensor() converts numpy (H, W, C) [0, 255] -> Tensor (C, H, W) [0.0, 1.0]
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float32)

        return img, label


def get_transforms(split):
    """
    Returns the transformations for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                # Cite solution_lesson_node_00002: Targeted augmentation for contrast robustness
                transforms.ColorJitter(brightness=0.2, contrast=0.3),
            ]
        )
    else:
        # For val and test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def create_dataloaders(
    batch_size=64,
    input_dir="./input",
    metadata_dir="./metadata",
    load_cached_data=True,
    num_workers=2,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        input_dir (str): Directory containing the image folders.
        metadata_dir (str): Directory containing metadata CSVs.
        load_cached_data (bool): Whether to use cached numpy arrays.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    # 1. Load Data
    train_ids, train_imgs, train_lbls = load_and_preprocess_data(
        os.path.join(metadata_dir, "train_metadata.csv"),
        input_dir,
        "train",
        load_cached_data,
    )
    val_ids, val_imgs, val_lbls = load_and_preprocess_data(
        os.path.join(metadata_dir, "val_metadata.csv"),
        input_dir,
        "val",
        load_cached_data,
    )
    test_ids, test_imgs, test_lbls = load_and_preprocess_data(
        os.path.join(metadata_dir, "test_metadata.csv"),
        input_dir,
        "test",
        load_cached_data,
    )

    # 2. Define Transforms
    train_transform = get_transforms("train")
    eval_transform = get_transforms("val")

    # 3. Create Datasets
    train_dataset = CactusDataset(train_imgs, train_lbls, transform=train_transform)
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=eval_transform)
    test_dataset = CactusDataset(test_imgs, test_lbls, transform=eval_transform)

    # 4. Create Loaders
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

    return train_loader, val_loader, test_loader, test_ids
