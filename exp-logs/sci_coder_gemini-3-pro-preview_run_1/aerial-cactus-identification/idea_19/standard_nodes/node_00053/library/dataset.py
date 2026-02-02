import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def load_or_process_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from cache if available and requested; otherwise processes from scratch
    and saves to cache.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, file_sizes, ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    fs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_fsizes.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(fs_cache_path)
            and os.path.exists(id_cache_path)
        ):
            # print(f"Loading {cache_prefix} data from cache...")
            images = np.load(img_cache_path)
            labels = np.load(lbl_cache_path)
            file_sizes = np.load(fs_cache_path)
            ids = np.load(id_cache_path, allow_pickle=True)
            return images, labels, file_sizes, ids

    # 2. Process from scratch
    # print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    img_list = []
    lbl_list = []
    fs_list = []
    id_list = []

    # Pre-calculate full paths
    # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
    file_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).tolist()
    )
    has_cactus = df["has_cactus"].values
    img_ids = df["id"].values

    for idx, fpath in enumerate(file_paths):
        if not os.path.exists(fpath):
            continue

        # Read Image
        # cv2 reads in BGR by default
        img = cv2.imread(fpath)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get file size in bytes
        fsize = os.path.getsize(fpath)

        img_list.append(img)
        lbl_list.append(has_cactus[idx])
        fs_list.append(fsize)
        id_list.append(img_ids[idx])

    images = np.array(img_list, dtype=np.uint8)
    labels = np.array(lbl_list, dtype=np.float32)
    file_sizes = np.array(fs_list, dtype=np.float32)
    ids = np.array(id_list)

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(fs_cache_path, file_sizes)
    np.save(id_cache_path, ids)

    return images, labels, file_sizes, ids


class CactusDataset(Dataset):
    """
    Dataset class that handles on-the-fly view generation (Structural, Chromatic, Holistic)
    and metadata normalization.
    """

    def __init__(
        self, images, labels, file_sizes, ids, mode, transform=None, fsize_stats=None
    ):
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.ids = ids
        self.mode = mode
        self.transform = transform

        # Metadata statistics for normalization (mean, std)
        self.fsize_mean = fsize_stats["mean"] if fsize_stats else 0.0
        self.fsize_std = fsize_stats["std"] if fsize_stats else 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve raw data
        image = self.images[idx]  # RGB uint8 (32, 32, 3)
        label = self.labels[idx]
        fsize = self.file_sizes[idx]

        # Apply Geometric Augmentations (Flip, Rotate, etc.)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented[
                "image"
            ]  # Still numpy array or tensor depending on transform
            # If transform returns TensorV2, it's (C, H, W). If numpy, (H, W, C).
            # We assume albumentations without ToTensorV2 here to keep it numpy for cv2 ops below
            # or we handle tensor conversion carefully.
            # Strategy: Use Albumentations that returns numpy, process views, then to Tensor.

        # Process Views based on Mode
        # Image is (H, W, 3) RGB uint8

        if self.mode == "structural":
            # View: L channel + Laplacian
            # 1. Convert to LAB and extract L
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]  # (32, 32) Range [0, 255]

            # 2. Compute Laplacian on Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)  # Float, can be negative

            # Normalize
            # L: [0, 255] -> [0, 1]
            l_norm = l_channel.astype(np.float32) / 255.0
            # Lap: Scale down to roughly unit range.
            # Laplacian of [0,255] image typically in [-1000, 1000]. /255 puts it in [-4, 4].
            lap_norm = laplacian / 255.0

            # Stack: (2, 32, 32)
            final_img = np.stack([l_norm, lap_norm], axis=0)

        elif self.mode == "chromatic":
            # View: A + B channels
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            a_channel = lab[:, :, 1]  # [0, 255]
            b_channel = lab[:, :, 2]  # [0, 255]

            # Normalize [0, 255] -> [0, 1]
            a_norm = a_channel.astype(np.float32) / 255.0
            b_norm = b_channel.astype(np.float32) / 255.0

            # Stack: (2, 32, 32)
            final_img = np.stack([a_norm, b_norm], axis=0)

        else:  # 'holistic' or default (RGB)
            # Normalize [0, 255] -> [0, 1]
            final_img = image.astype(np.float32) / 255.0
            # Transpose to (C, H, W) -> (3, 32, 32)
            final_img = final_img.transpose(2, 0, 1)

        # Convert to Tensor
        image_tensor = torch.from_numpy(final_img).float()

        # Process Metadata
        # 1. Normalized for FiLM (Z-score)
        fsize_norm = (fsize - self.fsize_mean) / (self.fsize_std + 1e-6)
        fsize_norm = torch.tensor([fsize_norm], dtype=torch.float32)

        # 2. Log-transformed for MTL Target
        fsize_log = np.log1p(fsize)
        fsize_target = torch.tensor([fsize_log], dtype=torch.float32)

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.float32),
            fsize_norm,
            fsize_target,
        )


def mixup_data(x, meta, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to inputs (image + metadata) and targets.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_meta = lam * meta + (1 - lam) * meta[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, mixed_meta, y_a, y_b, lam


def get_loaders(fold_idx, load_cached_data=True, model_name=Config.MODEL_HOLISTIC):
    """
    Prepares DataLoaders for training and validation.

    Args:
        fold_idx (int): Current fold index (0 to N_FOLDS-1). Not used for splitting here
                        as we use fixed validation set from metadata, but kept for API consistency.
        load_cached_data (bool): Use cached .npy files.
        model_name (str): Name of the model to determine the data mode.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Determine Mode from Model Name
    if model_name == Config.MODEL_STRUCTURAL:
        mode = "structural"
    elif model_name == Config.MODEL_CHROMATIC:
        mode = "chromatic"
    else:
        mode = "holistic"

    # Load Data
    # We load all subsets.
    # Note: The provided metadata already splits Train (80%) and Val (20%).
    # We will use these fixed splits.

    # Train
    train_imgs, train_lbls, train_fs, train_ids = load_or_process_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )

    # Val
    val_imgs, val_lbls, val_fs, val_ids = load_or_process_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )

    # Test
    test_imgs, test_lbls, test_fs, test_ids = load_or_process_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Debug Mode
    if Config.DEBUG:
        train_imgs = train_imgs[: Config.DEBUG_SAMPLE_SIZE]
        train_lbls = train_lbls[: Config.DEBUG_SAMPLE_SIZE]
        train_fs = train_fs[: Config.DEBUG_SAMPLE_SIZE]
        train_ids = train_ids[: Config.DEBUG_SAMPLE_SIZE]

    # Calculate File Size Statistics on Training Set
    fsize_mean = np.mean(train_fs)
    fsize_std = np.std(train_fs)
    fsize_stats = {"mean": fsize_mean, "std": fsize_std}

    # Augmentations
    # Only geometric augmentations here. Normalization/ToTensor handled in Dataset.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.5, border_mode=cv2.BORDER_REFLECT),
        ]
    )

    # No TTA in validation loader, pure validation
    val_transform = None
    test_transform = None

    # Datasets
    train_dataset = CactusDataset(
        train_imgs,
        train_lbls,
        train_fs,
        train_ids,
        mode=mode,
        transform=train_transform,
        fsize_stats=fsize_stats,
    )

    val_dataset = CactusDataset(
        val_imgs,
        val_lbls,
        val_fs,
        val_ids,
        mode=mode,
        transform=val_transform,
        fsize_stats=fsize_stats,
    )

    test_dataset = CactusDataset(
        test_imgs,
        test_lbls,
        test_fs,
        test_ids,
        mode=mode,
        transform=test_transform,
        fsize_stats=fsize_stats,
    )

    # Loaders
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
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
