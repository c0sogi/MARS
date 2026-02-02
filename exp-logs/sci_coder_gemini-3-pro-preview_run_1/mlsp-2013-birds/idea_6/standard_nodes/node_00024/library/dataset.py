import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    Custom Dataset for Bird Species Classification.
    Handles loading of processed spectrograms and labels.
    Implements the structural innovation of Spectral Deltas.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N, NumClasses).
            ids (np.ndarray, optional): Array of record IDs (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are already processed into (H, W, 3) float32 in the caching step
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback conversion if no transform provided
            image = torch.tensor(image).permute(2, 0, 1)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        elif self.ids is not None:
            rec_id = self.ids[idx]
            return image, rec_id
        else:
            return image

    @staticmethod
    def prepare_input(spec):
        """
        Replicates the single channel spectrogram to create a 3-channel input.
        This preserves natural image statistics better than engineered features
        for transfer learning (Cite solution_lesson_node_00022).

        Args:
            spec (np.ndarray): Single channel spectrogram (H, W).

        Returns:
            np.ndarray: 3-channel image (H, W, 3).
        """
        # Ensure input is float
        spec = spec.astype(np.float32)

        # Replicate channels: (H, W) -> (H, W, 3)
        img_3c = np.stack([spec, spec, spec], axis=-1)

        return img_3c


def process_data(metadata_path):
    """
    Loads raw data and processes it (Resize + Channel Replication).
    Caching is removed to ensure data integrity (Cite solution_lesson_node_00018).

    Args:
        metadata_path (str): Path to the CSV file (train/val/test).

    Returns:
        tuple: (images, labels, ids)
    """
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    # Identify label columns
    target_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    label_cols = [c for c in target_cols if c in df.columns]

    for idx, row in df.iterrows():
        # Construct full path to spectrogram
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = os.path.splitext(wav_filename)[0] + ".bmp"
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        if not os.path.exists(bmp_path):
            continue

        # Load Image
        img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize
        img_resized = cv2.resize(
            img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

        # Normalize to 0-1 range
        img_norm = img_resized.astype(np.float32) / 255.0

        # Prepare 3-channel input (Replication)
        img_3c = BirdDataset.prepare_input(img_norm)

        img_list.append(img_3c)
        id_list.append(row["rec_id"])

        # Handle Labels
        if len(label_cols) > 0:
            lbls = row[label_cols].values.astype(np.float32)
            label_list.append(lbls)
        else:
            label_list.append(np.zeros(Config.NUM_CLASSES, dtype=np.float32))

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    ids = np.array(id_list, dtype=np.int64)

    return images, labels, ids


def get_transforms(stage="train"):
    """
    Returns Albumentations transforms for the specified stage.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if stage == "train":
        return A.Compose(
            [
                # SpecAugment-like masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_HEIGHT // 8,
                    max_width=Config.IMG_WIDTH // 8,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize and Convert
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def get_data_loaders(config):
    """
    Prepares and returns DataLoaders for Train, Val, and Test sets.

    Args:
        config (Config): Configuration class.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(config.SEED)

    # --- 1. Load Data ---

    # Train Data (Fold 0)
    train_images, train_labels, _ = process_data(config.TRAIN_CSV)

    # Val Data (Fold 0)
    val_images, val_labels, _ = process_data(config.VAL_CSV)

    # Test Data (Fold 1)
    test_images, _, test_ids = process_data(config.TEST_CSV)

    # --- 2. Create Datasets ---

    # Debugging Subset
    if config.DEBUG:
        subset_size = min(config.DEBUG_SUBSET_SIZE, len(train_images))
        train_images = train_images[:subset_size]
        train_labels = train_labels[:subset_size]
        val_images = val_images[:subset_size]
        val_labels = val_labels[:subset_size]
        test_images = test_images[:subset_size]
        test_ids = test_ids[:subset_size]
        print(f"DEBUG MODE: Reduced training size to {subset_size}")

    train_dataset = BirdDataset(
        train_images, train_labels, transform=get_transforms("train")
    )

    val_dataset = BirdDataset(val_images, val_labels, transform=get_transforms("val"))

    test_dataset = BirdDataset(
        test_images, ids=test_ids, transform=get_transforms("test")
    )

    # --- 4. Create Loaders ---

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
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
