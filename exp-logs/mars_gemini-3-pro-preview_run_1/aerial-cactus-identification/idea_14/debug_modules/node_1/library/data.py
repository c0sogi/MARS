import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger(name="data_module")


def load_and_cache_split(
    metadata_path: str,
    cache_img_path: str,
    cache_label_path: str,
    cache_filesize_path: str,
    input_dir: str,
    load_cached: bool = True,
):
    """
    Loads data from disk or cache.
    Returns:
        imgs (np.ndarray): Shape (N, 3, 32, 32), float32, range [0, 1]
        labels (np.ndarray): Shape (N,), float32
        filesizes (np.ndarray): Shape (N,), float32 (raw bytes)
        ids (np.ndarray): Shape (N,), string (filenames)
    """
    # 1. Try loading from cache
    if load_cached:
        if (
            os.path.exists(cache_img_path)
            and os.path.exists(cache_label_path)
            and os.path.exists(cache_filesize_path)
        ):

            logger.info(
                f"Loading cached data from {os.path.dirname(cache_img_path)}..."
            )
            try:
                imgs = np.load(cache_img_path)
                labels = np.load(cache_label_path)
                filesizes = np.load(cache_filesize_path)
                # IDs are not strictly cached as npy usually, but we can reconstruct or load if needed.
                # For this pipeline, we primarily need imgs/labels/sizes for training.
                # However, for test set, we need IDs. We'll handle IDs via metadata df if needed,
                # or we can cache them. The Config defines CACHE_TEST_IDS.

                # If specific ID cache exists (mostly for test), load it
                cache_id_path = cache_img_path.replace("imgs.npy", "ids.npy")
                if os.path.exists(cache_id_path):
                    ids = np.load(cache_id_path, allow_pickle=True)
                else:
                    # Fallback to reading CSV just for IDs if strictly necessary,
                    # but usually we return what we have.
                    ids = None

                logger.info(f"Loaded {len(imgs)} samples from cache.")
                return imgs, labels, filesizes, ids
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    filesize_list = []
    id_list = []

    # Pre-allocate for speed if desired, but list append is fine for ~15k images
    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read Image
        if not os.path.exists(full_path):
            # Should not happen given metadata validation, but safety check
            continue

        # cv2 reads BGR
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        img = img.transpose(2, 0, 1)

        # Read File Size
        fsize = os.path.getsize(full_path)

        img_list.append(img)
        label_list.append(row["has_cactus"])
        filesize_list.append(fsize)
        id_list.append(row["id"])

    imgs = np.array(img_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    filesizes = np.array(filesize_list, dtype=np.float32)
    ids = np.array(id_list)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
    np.save(cache_img_path, imgs)
    np.save(cache_label_path, labels)
    np.save(cache_filesize_path, filesizes)

    # Save IDs if it's the test set or generally useful
    cache_id_path = cache_img_path.replace("imgs.npy", "ids.npy")
    np.save(cache_id_path, ids)

    logger.info(f"Processed and cached {len(imgs)} samples.")
    return imgs, labels, filesizes, ids


class CactusDataset(Dataset):
    """
    In-memory dataset for Cactus Identification.
    Returns (image, label, file_size_meta).
    """

    def __init__(self, images, labels, file_sizes, transform=None, ids=None):
        self.images = torch.from_numpy(images)  # (N, 3, 32, 32)
        self.labels = torch.from_numpy(labels).float()  # (N,)
        self.file_sizes = torch.from_numpy(file_sizes).float()  # (N,)
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        fsize = self.file_sizes[idx]

        # Apply geometric augmentations
        if self.transform:
            img = self.transform(img)

        # If IDs are present (Test set), one might want them, but standard training loop
        # expects (input, target, meta). We stick to the model signature.
        # For inference, the order is preserved, so we can map predictions back to IDs separately.

        return img, label, fsize


class MixupCollate:
    """
    Applies Mixup regularization to a batch.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: list of (img, label, fsize) tuples
        Returns:
            mixed_imgs, mixed_labels, fsizes
        """
        imgs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        fsizes = torch.tensor([item[2] for item in batch], dtype=torch.float32)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = imgs.size(0)
        index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[index, :]

        # For binary classification with BCEWithLogits, we mix the targets directly
        # label_a * lam + label_b * (1 - lam)
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        # We do NOT mix file sizes (metadata).
        # The metadata is used to modulate features based on the *input* quality.
        # Since the image is a mix, the "quality" is ambiguous.
        # However, usually in Mixup with metadata, we either mix metadata or keep original.
        # Given the FiLM logic modulates based on signal quality/compression,
        # and we are mixing two images, keeping the metadata aligned with the dominant image
        # or mixing it is a choice. Mixing metadata (linear interpolation) is consistent with the image mix.
        mixed_fsizes = lam * fsizes + (1 - lam) * fsizes[index]

        # Reshape labels for BCEWithLogitsLoss: (N, 1)
        mixed_labels = mixed_labels.view(-1, 1)
        mixed_fsizes = mixed_fsizes.view(-1, 1)

        return mixed_imgs, mixed_labels, mixed_fsizes


def get_transforms(mode="train"):
    """
    Returns transforms. Since data is already tensor (C,H,W),
    we use torchvision.transforms operating on tensors.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        return None  # No TTA here, TTA is handled in inference loop if needed


def get_data_loaders(config: Config, load_cached: bool = True):
    """
    Orchestrates loading, normalization, and DataLoader creation.
    """
    # 1. Load Data
    # Train
    train_imgs, train_labels, train_fs, _ = load_and_cache_split(
        config.TRAIN_METADATA_PATH,
        config.CACHE_TRAIN_IMGS,
        config.CACHE_TRAIN_LABELS,
        config.CACHE_TRAIN_FILESIZES,
        config.INPUT_DIR,
        load_cached,
    )

    # Val
    val_imgs, val_labels, val_fs, _ = load_and_cache_split(
        config.VAL_METADATA_PATH,
        config.CACHE_VAL_IMGS,
        config.CACHE_VAL_LABELS,
        config.CACHE_VAL_FILESIZES,
        config.INPUT_DIR,
        load_cached,
    )

    # Test
    test_imgs, test_labels, test_fs, test_ids = load_and_cache_split(
        config.TEST_METADATA_PATH,
        config.CACHE_TEST_IMGS,
        config.CACHE_TEST_IDS,  # We save labels here just to satisfy signature, but we need IDs
        config.CACHE_TEST_FILESIZES,
        config.INPUT_DIR,
        load_cached,
    )
    # Note: load_and_cache_split saves 'labels' to the 2nd arg.
    # For test, we passed CACHE_TEST_IDS as the 2nd arg path for labels?
    # Wait, the function signature is (meta, cache_img, cache_label, cache_fsize).
    # Config has CACHE_TEST_IDS. The test metadata has 'has_cactus' column (0.5).
    # So we should pass a path for test labels (dummy) or just reuse a temp one.
    # Actually, let's fix the call:
    # We need to store test IDs separately. load_and_cache_split returns 4 items.
    # The cache_label_path argument is where labels are saved.
    # Config doesn't explicitly have CACHE_TEST_LABELS, but we can infer or add it.
    # Let's just use a derived path.

    # Re-calling Test with correct paths
    test_labels_cache = config.CACHE_TEST_IDS.replace("ids.npy", "labels.npy")
    test_imgs, test_labels, test_fs, test_ids = load_and_cache_split(
        config.TEST_METADATA_PATH,
        config.CACHE_TEST_IMGS,
        test_labels_cache,
        config.CACHE_TEST_FILESIZES,
        config.INPUT_DIR,
        load_cached,
    )

    # 2. Normalize File Sizes (Z-score based on TRAIN stats)
    fs_mean = np.mean(train_fs)
    fs_std = np.std(train_fs) + 1e-8  # Avoid div by zero

    logger.info(f"File Size Stats (Train) - Mean: {fs_mean:.4f}, Std: {fs_std:.4f}")

    train_fs_norm = (train_fs - fs_mean) / fs_std
    val_fs_norm = (val_fs - fs_mean) / fs_std
    test_fs_norm = (test_fs - fs_mean) / fs_std

    # 3. Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_labels, train_fs_norm, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(
        val_imgs, val_labels, val_fs_norm, transform=get_transforms("val")
    )

    test_dataset = CactusDataset(
        test_imgs,
        test_labels,
        test_fs_norm,
        transform=get_transforms("test"),  # No aug by default
        ids=test_ids,
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=MixupCollate(alpha=config.MIXUP_ALPHA),
        pin_memory=True,
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
