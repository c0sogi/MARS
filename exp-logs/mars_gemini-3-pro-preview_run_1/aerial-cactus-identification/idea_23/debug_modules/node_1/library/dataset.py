import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    def __init__(
        self, metadata_path, img_dir=None, mode="train", load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            img_dir (str, optional): Root directory of images. Defaults to Config.INPUT_DIR.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and cache naming.
            load_cached_data (bool): Whether to try loading from cache.
        """
        self.mode = mode
        self.metadata_path = metadata_path
        self.img_dir = img_dir if img_dir else Config.INPUT_DIR

        # Define cache paths based on mode to separate train/val/test artifacts
        self.cache_dir = Config.CACHE_DIR
        self.cache_imgs_path = os.path.join(self.cache_dir, f"{mode}_imgs.npy")
        self.cache_labels_path = os.path.join(self.cache_dir, f"{mode}_labels.npy")
        self.cache_quality_path = os.path.join(self.cache_dir, f"{mode}_quality.npy")

        # Define Augmentations for Training
        if self.mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        else:
            self.transform = None

        # Load Data (either from cache or process from scratch)
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads data from cache if available and requested, otherwise processes from source.
        """
        cache_exists = (
            os.path.exists(self.cache_imgs_path)
            and os.path.exists(self.cache_labels_path)
            and os.path.exists(self.cache_quality_path)
        )

        if load_cached_data and cache_exists:
            print(f"Loading cached {self.mode} data from {self.cache_dir}...")
            self.images = np.load(self.cache_imgs_path)
            self.labels = np.load(self.cache_labels_path)
            self.quality_targets = np.load(self.cache_quality_path)
        else:
            print(f"Processing {self.mode} data from scratch...")
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads images and metadata, processes them, and saves to cache.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"Debug mode enabled: Limiting {self.mode} dataset to 100 samples.")
            df = df.head(100)

        img_list = []
        label_list = []
        file_sizes = []

        # Iterate through metadata
        for _, row in df.iterrows():
            # Construct full path
            # metadata 'file_path' is relative to input dir (e.g., "train/id.jpg")
            rel_path = row["file_path"]
            full_path = os.path.join(self.img_dir, rel_path)

            # 1. Load and Preprocess Image
            img = cv2.imread(full_path)
            if img is None:
                # In case of missing file (though metadata validation should catch this),
                # create a blank image to avoid crashing.
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to 0-1 float32
            img = img.astype(np.float32) / 255.0

            # Transpose to (C, H, W) format for PyTorch
            img = np.transpose(img, (2, 0, 1))

            img_list.append(img)

            # 2. Extract Label
            # 'has_cactus' is 0 or 1 for train/val, and placeholder 0.5 for test
            label_list.append(row["has_cactus"])

            # 3. Extract File Size for Auxiliary Task
            try:
                fsize = os.path.getsize(full_path)
            except OSError:
                fsize = 0
            file_sizes.append(fsize)

        # Convert lists to numpy arrays
        self.images = np.array(img_list, dtype=np.float32)
        self.labels = np.array(label_list, dtype=np.float32).reshape(-1, 1)

        # Process Quality Targets (File Sizes)
        # Log transform to reduce skewness
        log_sizes = np.log1p(np.array(file_sizes, dtype=np.float32))

        # Normalize to 0-1 range based on the current split's statistics
        min_s = log_sizes.min()
        max_s = log_sizes.max()
        denom = max_s - min_s
        if denom == 0:
            denom = 1.0

        self.quality_targets = (log_sizes - min_s) / denom
        self.quality_targets = self.quality_targets.reshape(-1, 1)

        # Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.cache_imgs_path, self.images)
        np.save(self.cache_labels_path, self.labels)
        np.save(self.cache_quality_path, self.quality_targets)

        print(
            f"Successfully cached {len(self.images)} {self.mode} samples to {self.cache_dir}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            img_tensor (torch.Tensor): Image tensor (C, H, W)
            label_tensor (torch.Tensor): Class label (1, )
            quality_tensor (torch.Tensor): Quality target (1, )
        """
        # Retrieve data from RAM
        img = self.images[idx]
        label = self.labels[idx]
        quality = self.quality_targets[idx]

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label)
        quality_tensor = torch.from_numpy(quality)

        # Apply augmentations (only affects image)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor, quality_tensor
