import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the given mode.
    Note: Resize and CenterCrop are applied during the data processing/caching phase.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # val or test
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def _resize_and_center_crop(img, resize_dim, crop_dim):
    """
    Resizes the image such that the smaller edge is resize_dim,
    then performs a center crop of crop_dim.
    """
    h, w = img.shape[:2]

    # Resize logic to match torchvision.transforms.Resize(size) behavior
    # Resize the smaller edge to resize_dim, maintaining aspect ratio
    if h < w:
        new_h = resize_dim
        new_w = int(w * (resize_dim / h))
    else:
        new_w = resize_dim
        new_h = int(h * (resize_dim / w))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Center Crop
    h_r, w_r = img_resized.shape[:2]
    top = (h_r - crop_dim) // 2
    left = (w_r - crop_dim) // 2

    # Ensure indices are valid
    top = max(0, top)
    left = max(0, left)

    img_cropped = img_resized[top : top + crop_dim, left : left + crop_dim]

    return img_cropped


def _process_data(load_cached_data=True):
    """
    Loads data from metadata, processes images (Resize+Crop), and caches them as .npy files.
    Returns labeled data (images, targets, ids) and test data (images, ids), plus label map.
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Cache file paths
    cache_train_imgs = os.path.join(Config.WORK_DIR, "cache_train_imgs.npy")
    cache_train_targets = os.path.join(Config.WORK_DIR, "cache_train_targets.npy")
    cache_train_ids = os.path.join(Config.WORK_DIR, "cache_train_ids.npy")
    cache_test_imgs = os.path.join(Config.WORK_DIR, "cache_test_imgs.npy")
    cache_test_ids = os.path.join(Config.WORK_DIR, "cache_test_ids.npy")
    cache_label_map = os.path.join(Config.WORK_DIR, "label_map.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(cache_train_imgs)
        and os.path.exists(cache_train_targets)
        and os.path.exists(cache_train_ids)
        and os.path.exists(cache_test_imgs)
        and os.path.exists(cache_test_ids)
        and os.path.exists(cache_label_map)
    ):

        print("Loading cached data from", Config.WORK_DIR)
        train_imgs = np.load(cache_train_imgs)
        train_targets = np.load(cache_train_targets)
        train_ids = np.load(cache_train_ids)
        test_imgs = np.load(cache_test_imgs)
        test_ids = np.load(cache_test_ids)
        label_map = np.load(cache_label_map, allow_pickle=True).item()

        return train_imgs, train_targets, train_ids, test_imgs, test_ids, label_map

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Combine train and val for Stratified K-Fold
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # 2. Create Label Map
    unique_breeds = sorted(full_train_df["breed"].unique())
    label_map = {breed: idx for idx, breed in enumerate(unique_breeds)}

    # 3. Process Labeled Images
    train_imgs_list = []
    train_targets_list = []
    train_ids_list = []

    for idx, row in full_train_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and Crop
        img = _resize_and_center_crop(img, Config.IMG_SIZE, Config.CROP_SIZE)

        train_imgs_list.append(img)
        train_targets_list.append(label_map[row["breed"]])
        train_ids_list.append(row["id"])

    train_imgs = np.array(train_imgs_list, dtype=np.uint8)
    train_targets = np.array(train_targets_list, dtype=np.int64)
    train_ids = np.array(train_ids_list)

    # 4. Process Test Images
    test_imgs_list = []
    test_ids_list = []

    for idx, row in test_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _resize_and_center_crop(img, Config.IMG_SIZE, Config.CROP_SIZE)

        test_imgs_list.append(img)
        test_ids_list.append(row["id"])

    test_imgs = np.array(test_imgs_list, dtype=np.uint8)
    test_ids = np.array(test_ids_list)

    # 5. Save to Cache
    np.save(cache_train_imgs, train_imgs)
    np.save(cache_train_targets, train_targets)
    np.save(cache_train_ids, train_ids)
    np.save(cache_test_imgs, test_imgs)
    np.save(cache_test_ids, test_ids)
    np.save(cache_label_map, label_map)

    print(
        f"Data processed and cached. Train shape: {train_imgs.shape}, Test shape: {test_imgs.shape}"
    )

    return train_imgs, train_targets, train_ids, test_imgs, test_ids, label_map


class DogDataset(Dataset):
    def __init__(self, images, targets=None, ids=None, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3)
            targets (np.ndarray, optional): Array of labels (N,)
            ids (np.ndarray, optional): Array of IDs (N,)
            transforms (callable, optional): Transformations to apply
        """
        self.images = images
        self.targets = targets
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        # Handle target
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = -1  # Dummy target for test set

        # Handle ID
        if self.ids is not None:
            img_id = self.ids[idx]
        else:
            img_id = ""

        return img, target, img_id


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates train and validation dataloaders for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    # Load all data
    imgs, targets, ids, _, _, _ = _process_data(load_cached_data)

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(imgs, targets))
    if fold_idx < 0 or fold_idx >= Config.N_FOLDS:
        raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.N_FOLDS-1})")

    train_idx, val_idx = splits[fold_idx]

    # Create subsets
    train_imgs, val_imgs = imgs[train_idx], imgs[val_idx]
    train_targets, val_targets = targets[train_idx], targets[val_idx]
    train_ids, val_ids = ids[train_idx], ids[val_idx]

    # Create Datasets
    train_dataset = DogDataset(
        images=train_imgs,
        targets=train_targets,
        ids=train_ids,
        transforms=get_transforms(mode="train"),
    )

    val_dataset = DogDataset(
        images=val_imgs,
        targets=val_targets,
        ids=val_ids,
        transforms=get_transforms(mode="val"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates the test dataloader.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        test_loader
    """
    _, _, _, test_imgs, test_ids, _ = _process_data(load_cached_data)

    test_dataset = DogDataset(
        images=test_imgs,
        targets=None,
        ids=test_ids,
        transforms=get_transforms(mode="test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader


def get_label_map(load_cached_data=True):
    """
    Returns the dictionary mapping breed names to integer indices.
    """
    _, _, _, _, _, label_map = _process_data(load_cached_data)
    return label_map
