import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything

# Define ImageNet stats for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CactusDataset(Dataset):
    def __init__(self, images, labels, file_sizes, ids, transform=None, phase="train"):
        """
        Args:
            images (np.ndarray): Shape (N, 3, H, W), float32 in [0, 1].
            labels (np.ndarray): Shape (N,), float32.
            file_sizes (np.ndarray): Shape (N,), float32 normalized log sizes.
            ids (np.ndarray): Shape (N,), string IDs.
            transform (callable, optional): Optional transform to be applied on a sample.
            phase (str): 'train', 'valid', or 'test'.
        """
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.ids = ids
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images are already (C, H, W) float32 tensors in [0, 1]
        img = torch.tensor(self.images[idx], dtype=torch.float32)

        # Apply geometric transforms (flips) if any
        if self.transform:
            img = self.transform(img)

        # Apply normalization (standard ImageNet)
        normalizer = transforms.Normalize(mean=MEAN, std=STD)
        img = normalizer(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        fsize = torch.tensor(self.file_sizes[idx], dtype=torch.float32)
        img_id = self.ids[idx]

        return img, label, fsize, img_id


def get_transforms(phase="train"):
    """
    Returns the transformations for the given phase.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # No geometric transforms for validation/test
        # Normalization is handled in Dataset
        return None


def mixup_data(x, y_class, y_size, alpha=0.2, device=Config.DEVICE):
    """
    Applies Mixup to inputs and targets.
    Returns:
        mixed_x: Mixed images
        mixed_y_class: Mixed class labels
        mixed_y_size: Mixed regression targets
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Mix targets directly
    mixed_y_class = lam * y_class + (1 - lam) * y_class[index]
    mixed_y_size = lam * y_size + (1 - lam) * y_size[index]

    return mixed_x, mixed_y_class, mixed_y_size


def _process_split(metadata_path, input_dir, is_test=False):
    """
    Helper to process a metadata CSV and load images.
    """
    df = pd.read_csv(metadata_path)

    imgs = []
    labels = []
    fsizes = []
    ids = []

    for _, row in df.iterrows():
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read Image (CV2 reads as BGR)
        img_bgr = cv2.imread(full_path)

        if img_bgr is None:
            # Handle missing file gracefully by creating a black image
            img_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            fsize = 0
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if os.path.exists(full_path):
                fsize = os.path.getsize(full_path)
            else:
                fsize = 0

        # Ensure 32x32
        if img_rgb.shape[0] != 32 or img_rgb.shape[1] != 32:
            img_rgb = cv2.resize(img_rgb, (32, 32))

        # Convert to float32 [0, 1] and CHW format
        img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        imgs.append(img_tensor)
        fsizes.append(fsize)
        ids.append(img_id)

        if is_test:
            labels.append(0.5)  # Placeholder
        else:
            labels.append(row["has_cactus"])

    return (
        np.array(imgs),
        np.array(labels, dtype=np.float32),
        np.array(fsizes, dtype=np.float32),
        np.array(ids),
    )


def cache_dataset_in_ram(load_cached_data=True):
    """
    Loads, processes, and caches the dataset in RAM.
    Returns:
        (train_data, val_data, test_data) where each is a tuple of (imgs, labels, fsizes, ids)
    """
    Config.setup()

    # Define local cache paths for validation and missing test labels
    cache_val_imgs = os.path.join(Config.CACHE_DIR, "val_imgs.npy")
    cache_val_labels = os.path.join(Config.CACHE_DIR, "val_labels.npy")
    cache_val_fsizes = os.path.join(Config.CACHE_DIR, "val_fsizes.npy")
    cache_val_ids = os.path.join(Config.CACHE_DIR, "val_ids.npy")
    cache_test_labels = os.path.join(Config.CACHE_DIR, "test_labels.npy")

    # Check if all cache files exist
    required_files = [
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TRAIN_FSIZES,
        Config.CACHE_TRAIN_IDS,
        cache_val_imgs,
        cache_val_labels,
        cache_val_fsizes,
        cache_val_ids,
        Config.CACHE_TEST_IMGS,
        cache_test_labels,
        Config.CACHE_TEST_FSIZES,
        Config.CACHE_TEST_IDS,
    ]

    files_exist = all(os.path.exists(f) for f in required_files)

    if load_cached_data and files_exist:
        print("Loading cached data from disk...")
        train_data = (
            np.load(Config.CACHE_TRAIN_IMGS),
            np.load(Config.CACHE_TRAIN_LABELS),
            np.load(Config.CACHE_TRAIN_FSIZES),
            np.load(Config.CACHE_TRAIN_IDS),
        )
        val_data = (
            np.load(cache_val_imgs),
            np.load(cache_val_labels),
            np.load(cache_val_fsizes),
            np.load(cache_val_ids),
        )
        test_data = (
            np.load(Config.CACHE_TEST_IMGS),
            np.load(cache_test_labels),
            np.load(Config.CACHE_TEST_FSIZES),
            np.load(Config.CACHE_TEST_IDS),
        )
        return train_data, val_data, test_data

    print("Processing dataset from scratch...")

    # Process splits
    tr_imgs, tr_lbls, tr_fs_raw, tr_ids = _process_split(
        Config.TRAIN_META_PATH, Config.INPUT_DIR, is_test=False
    )
    val_imgs, val_lbls, val_fs_raw, val_ids = _process_split(
        Config.VAL_META_PATH, Config.INPUT_DIR, is_test=False
    )
    te_imgs, te_lbls, te_fs_raw, te_ids = _process_split(
        Config.TEST_META_PATH, Config.INPUT_DIR, is_test=True
    )

    # Normalize File Sizes
    # 1. Log transform (log1p to handle 0 safely)
    tr_fs_log = np.log1p(tr_fs_raw)
    val_fs_log = np.log1p(val_fs_raw)
    te_fs_log = np.log1p(te_fs_raw)

    # 2. Min-Max Normalize using Train stats
    min_val = tr_fs_log.min()
    max_val = tr_fs_log.max()
    denom = max_val - min_val if max_val > min_val else 1.0

    tr_fs = (tr_fs_log - min_val) / denom
    val_fs = (val_fs_log - min_val) / denom
    te_fs = (te_fs_log - min_val) / denom

    # Clip to [0, 1]
    tr_fs = np.clip(tr_fs, 0.0, 1.0)
    val_fs = np.clip(val_fs, 0.0, 1.0)
    te_fs = np.clip(te_fs, 0.0, 1.0)

    # Save to Cache
    np.save(Config.CACHE_TRAIN_IMGS, tr_imgs)
    np.save(Config.CACHE_TRAIN_LABELS, tr_lbls)
    np.save(Config.CACHE_TRAIN_FSIZES, tr_fs)
    np.save(Config.CACHE_TRAIN_IDS, tr_ids)

    np.save(cache_val_imgs, val_imgs)
    np.save(cache_val_labels, val_lbls)
    np.save(cache_val_fsizes, val_fs)
    np.save(cache_val_ids, val_ids)

    np.save(Config.CACHE_TEST_IMGS, te_imgs)
    np.save(cache_test_labels, te_lbls)
    np.save(Config.CACHE_TEST_FSIZES, te_fs)
    np.save(Config.CACHE_TEST_IDS, te_ids)

    return (
        (tr_imgs, tr_lbls, tr_fs, tr_ids),
        (val_imgs, val_lbls, val_fs, val_ids),
        (te_imgs, te_lbls, te_fs, te_ids),
    )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    (
        (tr_img, tr_lbl, tr_fs, tr_id),
        (val_img, val_lbl, val_fs, val_id),
        (te_img, te_lbl, te_fs, te_id),
    ) = cache_dataset_in_ram(load_cached_data)

    train_dataset = CactusDataset(
        tr_img, tr_lbl, tr_fs, tr_id, transform=get_transforms("train"), phase="train"
    )
    val_dataset = CactusDataset(
        val_img,
        val_lbl,
        val_fs,
        val_id,
        transform=get_transforms("valid"),
        phase="valid",
    )
    test_dataset = CactusDataset(
        te_img, te_lbl, te_fs, te_id, transform=get_transforms("test"), phase="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
