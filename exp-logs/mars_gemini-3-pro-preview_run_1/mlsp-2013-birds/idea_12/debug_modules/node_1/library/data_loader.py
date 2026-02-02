import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.configuration import Config
from library.utilities import set_seed


# ==========================================
# Augmentation Class
# ==========================================
class DeterministicAugmenter:
    """
    Applies Unstructured Cutout and Horizontal Flip deterministically
    using a seeded NumPy generator.
    """

    def __init__(self, seed, prob_flip=0.5, prob_cutout=0.5):
        self.rng = np.random.default_rng(seed)
        self.prob_flip = prob_flip
        self.prob_cutout = prob_cutout

    def reseed(self, seed):
        """Reseeds the generator. Used by worker_init_fn."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): Image of shape (H, W, C)
        Returns:
            np.ndarray: Augmented image
        """
        # Horizontal Flip
        if self.rng.random() < self.prob_flip:
            image = np.fliplr(image).copy()

        # Unstructured Cutout
        if self.rng.random() < self.prob_cutout:
            h, w, _ = image.shape
            # Random mask size (10% to 50% of dimension)
            mask_h = self.rng.integers(h // 10, h // 2)
            mask_w = self.rng.integers(w // 10, w // 2)

            # Random position
            y = self.rng.integers(0, h - mask_h)
            x = self.rng.integers(0, w - mask_w)

            image[y : y + mask_h, x : x + mask_w, :] = 0.0

        return image


# ==========================================
# Dataset Class
# ==========================================
class BirdDataset(Dataset):
    """
    Dynamic Bird Spectrogram Dataset.
    Loads BMPs, resizes to 256x640, and applies channel replication.
    """

    def __init__(self, df, config, mode="train", augmenter=None):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.augmenter = augmenter
        self.image_paths = []
        self.labels = []

        # Pre-process paths and labels
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.df.iterrows():
            # Convert wav path to spectrogram path
            # Metadata: essential_data/src_wavs/filename.wav
            # Target: supplemental_data/spectrograms/filename.bmp
            wav_rel_path = row["file_path"]
            basename = os.path.basename(wav_rel_path)
            bmp_name = os.path.splitext(basename)[0] + ".bmp"

            full_path = os.path.join(self.config.SPECTROGRAM_DIR, bmp_name)
            self.image_paths.append(full_path)

            if self.mode == "test":
                # Dummy labels for test set
                self.labels.append(np.zeros(self.config.NUM_CLASSES, dtype=np.float32))
            else:
                # Extract multi-hot labels
                label_cols = [f"species_{i}" for i in range(self.config.NUM_CLASSES)]
                # Ensure columns exist, else fallback (e.g. for pseudo-labels dataframe)
                if all(col in row for col in label_cols):
                    lbl = row[label_cols].values.astype(np.float32)
                else:
                    # If passed a dataframe without explicit species_x columns (unlikely with correct metadata)
                    lbl = np.zeros(self.config.NUM_CLASSES, dtype=np.float32)
                self.labels.append(lbl)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # 1. Dynamic Loading (Grayscale)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files gracefully (though EDA confirmed existence)
        if img is None:
            img = np.zeros(
                (self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8
            )

        # 2. High-Fidelity Resize (256 x 640)
        img = cv2.resize(
            img,
            (self.config.IMG_WIDTH, self.config.IMG_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # 3. Normalize and Channel Replication
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        # Stack to 3 channels (H, W, 3)
        img = np.stack([img, img, img], axis=-1)

        # 4. Augmentation
        if self.mode == "train" and self.augmenter is not None:
            img = self.augmenter(img)

        # 5. To Tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Return index for tracking/pseudo-labeling
        return img, label, idx


# ==========================================
# Worker Seeding
# ==========================================
def seed_worker(worker_id):
    """
    Seeds the worker and re-seeds the DeterministicAugmenter
    to ensure diversity across workers while maintaining determinism.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)

    # Re-seed the augmenter inside the dataset
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "augmenter") and dataset.augmenter is not None:
            dataset.augmenter.reseed(worker_seed)


# ==========================================
# Data Loading Functions
# ==========================================
def get_dataloaders(config, load_cached_data=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Flag for caching (kept for interface compliance).
                                 Metadata is loaded from CSVs directly.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Initialize Augmenter
    # Initial seed is from config. Workers will reseed this based on their ID.
    train_augmenter = DeterministicAugmenter(
        seed=config.SEED, prob_flip=config.PROB_FLIP, prob_cutout=config.PROB_CUTOUT
    )

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, config, mode="train", augmenter=train_augmenter
    )
    val_dataset = BirdDataset(val_df, config, mode="val")
    test_dataset = BirdDataset(test_df, config, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
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


def get_combined_dataloader(config, pseudo_labels_path):
    """
    Creates a DataLoader for the Student model by combining
    labeled training data and pseudo-labeled test data.

    Args:
        config (Config): Configuration object.
        pseudo_labels_path (str): Path to the parquet file containing pseudo-labels.

    Returns:
        DataLoader: Combined training loader.
    """
    # 1. Load Original Train Data
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)

    # 2. Load Pseudo-Labeled Test Data
    if os.path.exists(pseudo_labels_path):
        pseudo_df = pd.read_parquet(pseudo_labels_path)

        # Ensure columns match
        # train_df has 'rec_id', 'file_path', 'species_0'...'species_18'
        # pseudo_df should have the same.

        # Concatenate
        combined_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)
        print(
            f"Combined Dataset: {len(train_df)} labeled + {len(pseudo_df)} pseudo = {len(combined_df)} total."
        )
    else:
        print(
            f"Warning: Pseudo-labels file not found at {pseudo_labels_path}. Using only labeled data."
        )
        combined_df = train_df

    # 3. Initialize Augmenter
    augmenter = DeterministicAugmenter(
        seed=config.SEED, prob_flip=config.PROB_FLIP, prob_cutout=config.PROB_CUTOUT
    )

    # 4. Create Dataset
    combined_dataset = BirdDataset(
        combined_df, config, mode="train", augmenter=augmenter
    )

    # 5. Create DataLoader
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
    )

    return combined_loader
