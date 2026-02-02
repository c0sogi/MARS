import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    Handles images loaded as numpy arrays and applies transformations.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            # Return label as float32 for BCE loss
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Return dummy label for test set
            return image, torch.tensor(-1.0, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Gentle color jitter as per strategy
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Val and Test
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Performs Mixup data augmentation on the batch.

    Args:
        x (torch.Tensor): Input batch images.
        y (torch.Tensor): Input batch labels.
        alpha (float): Mixup alpha parameter.
        use_cuda (bool): Whether to use CUDA for index generation.

    Returns:
        mixed_x: Mixed input images.
        y_a: Labels of the first image set.
        y_b: Labels of the second image set.
        lam: Lambda value used for mixing.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion: The loss function.
        pred: Model predictions.
        y_a: Labels of the first image set.
        y_b: Labels of the second image set.
        lam: Lambda value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images, with caching mechanism.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
            images: np.ndarray of shape (N, H, W, C)
            labels: np.ndarray of shape (N,)
            ids: np.ndarray of shape (N,)
    """
    # Determine cache directory
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Adjust cache name if in debug mode to avoid polluting full cache
    if Config.DEBUG:
        cache_prefix = f"{cache_prefix}_debug"

    imgs_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    lbls_path = os.path.join(cache_dir, f"{cache_prefix}_lbls.npy")
    ids_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(imgs_path)
            and os.path.exists(lbls_path)
            and os.path.exists(ids_path)
        ):
            print(f"Loading cached data from {imgs_path}...")
            try:
                imgs = np.load(imgs_path)
                lbls = np.load(lbls_path)
                ids = np.load(ids_path)
                return imgs, lbls, ids
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    img_list = []
    lbl_list = []
    id_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {full_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        lbl_list.append(row["has_cactus"])
        id_list.append(row["id"])

    imgs = np.array(img_list)
    lbls = np.array(lbl_list)
    ids = np.array(id_list)

    # 3. Save to cache
    print(f"Saving cache to {cache_dir}...")
    np.save(imgs_path, imgs)
    np.save(lbls_path, lbls)
    np.save(ids_path, ids)

    return imgs, lbls, ids
