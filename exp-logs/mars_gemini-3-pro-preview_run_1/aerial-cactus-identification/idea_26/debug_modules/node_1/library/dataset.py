import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.utils import seed_everything

# Constants for Quality Target Normalization
# Based on typical file sizes: log(500) ~= 6.2, log(5000) ~= 8.5
LOG_SIZE_MIN = 6.0
LOG_SIZE_MAX = 9.0


class CactusDataset(Dataset):
    """
    Dataset class for Cactus Identification task with auxiliary quality supervision.
    Handles caching of images and targets to RAM/Disk to minimize I/O.
    """

    def __init__(
        self,
        metadata_file: str,
        input_dir: str = "./input",
        cache_dir: str = "./working/idea_26",
        split: str = "train",
        load_cached_data: bool = True,
        transform=None,
    ):
        """
        Args:
            metadata_file (str): Path to the metadata CSV file.
            input_dir (str): Root directory of the input data.
            cache_dir (str): Directory to store/load cached .npy files.
            split (str): Dataset split ('train', 'val', 'test').
            load_cached_data (bool): Whether to try loading from cache.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform
        self.input_dir = input_dir
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.cache_imgs_path = os.path.join(cache_dir, f"cache_{split}_imgs.npy")
        self.cache_labels_path = os.path.join(cache_dir, f"cache_{split}_labels.npy")
        self.cache_qual_path = os.path.join(cache_dir, f"cache_{split}_qual.npy")
        self.cache_ids_path = os.path.join(cache_dir, f"cache_{split}_ids.npy")

        # Load data
        if load_cached_data and self._check_cache_exists():
            print(f"Loading {split} data from cache: {cache_dir}")
            self.images = np.load(self.cache_imgs_path)
            self.labels = np.load(self.cache_labels_path)
            self.quality_targets = np.load(self.cache_qual_path)
            self.ids = np.load(self.cache_ids_path, allow_pickle=True)
        else:
            print(f"Processing {split} data from scratch...")
            self._process_and_cache(metadata_file)

    def _check_cache_exists(self) -> bool:
        return (
            os.path.exists(self.cache_imgs_path)
            and os.path.exists(self.cache_labels_path)
            and os.path.exists(self.cache_qual_path)
            and os.path.exists(self.cache_ids_path)
        )

    def _process_and_cache(self, metadata_file: str):
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pd.read_csv(metadata_file)

        img_list = []
        label_list = []
        qual_list = []
        id_list = []

        # Pre-allocate lists to avoid dynamic resizing overhead if possible,
        # but simple append is fine for this dataset size (~14k).

        for _, row in df.iterrows():
            img_id = row["id"]
            rel_path = row["file_path"]
            label = row["has_cactus"]

            full_path = os.path.join(self.input_dir, rel_path)

            # 1. Read Image
            if not os.path.exists(full_path):
                # Should not happen based on metadata validation, but safety check
                print(f"Warning: Image not found {full_path}")
                continue

            img = cv2.imread(full_path)
            if img is None:
                print(f"Warning: Failed to load image {full_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to 0-1 float32
            img = img.astype(np.float32) / 255.0

            # Transpose to (C, H, W) for PyTorch
            img = np.transpose(img, (2, 0, 1))

            # 2. Extract Quality Target (File Size)
            file_size = os.path.getsize(full_path)
            # Log transform
            log_size = np.log(float(file_size))
            # Normalize to 0-1 range
            norm_qual = (log_size - LOG_SIZE_MIN) / (LOG_SIZE_MAX - LOG_SIZE_MIN)
            # Clip to ensure bounds
            norm_qual = np.clip(norm_qual, 0.0, 1.0)

            img_list.append(img)
            label_list.append(float(label))
            qual_list.append(norm_qual)
            id_list.append(img_id)

        # Convert to numpy arrays
        self.images = np.array(img_list, dtype=np.float32)
        self.labels = np.array(label_list, dtype=np.float32)
        self.quality_targets = np.array(qual_list, dtype=np.float32)
        self.ids = np.array(id_list)

        # Save to cache
        print(f"Saving {self.split} cache to {self.cache_dir}...")
        np.save(self.cache_imgs_path, self.images)
        np.save(self.cache_labels_path, self.labels)
        np.save(self.cache_qual_path, self.quality_targets)
        np.save(self.cache_ids_path, self.ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx]  # (C, H, W)
        label = self.labels[idx]
        quality = self.quality_targets[idx]

        # Apply Geometric Augmentations (Train only)
        # Using numpy operations on the tensor-like array
        if self.split == "train":
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]  # Flip W

            # Random Vertical Flip
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]  # Flip H

            # Copy to avoid negative stride issues in torch
            img = img.copy()

        # Convert to Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        qual_tensor = torch.tensor(quality, dtype=torch.float32)

        # Note: We do not return ID in training loop for speed,
        # but for test/inference pipelines, one might need to access .ids[idx] separately.

        return img_tensor, label_tensor, qual_tensor

    def get_ids(self):
        return self.ids


def mixup_data(x, y_cls, y_qual, alpha=0.2, device="cuda"):
    """
    Applies Mixup regularization to inputs and both targets (classification and quality).

    Args:
        x (torch.Tensor): Input images.
        y_cls (torch.Tensor): Classification labels.
        y_qual (torch.Tensor): Quality regression targets.
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform calculation on.

    Returns:
        mixed_x, y_cls_a, y_cls_b, y_qual_a, y_qual_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_cls_a, y_cls_b = y_cls, y_cls[index]
    y_qual_a, y_qual_b = y_qual, y_qual[index]

    return mixed_x, y_cls_a, y_cls_b, y_qual_a, y_qual_b, lam
