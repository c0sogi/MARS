import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles dynamic loading of spectrograms, resizing, channel replication,
    normalization, and augmentation (SpecAugment).
    Supports both hard labels (training) and soft labels (distillation).
    """

    def __init__(self, df, mode="train", soft_labels=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (rec_id, file_path, species columns).
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label return.
            soft_labels (np.ndarray, optional): Soft labels for distillation.
                                                Shape (N, NUM_CLASSES). If provided, these are used as targets.
                                                Must align with the dataframe indices.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.soft_labels = soft_labels

        # Identify label columns for hard labels
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-compute full image paths to avoid overhead in __getitem__
        # Maps wav filename in metadata to bmp filename in spectrogram directory
        self.image_paths = []
        for _, row in self.df.iterrows():
            wav_rel_path = row["file_path"]
            # Extract filename, e.g., "PC10_... .wav"
            basename = os.path.basename(wav_rel_path)
            # Change extension to .bmp
            bmp_name = os.path.splitext(basename)[0] + ".bmp"
            # Construct full path
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)
            self.image_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Dynamic In-Memory Loading
        img_path = self.image_paths[idx]

        # Robust loading: handle potential missing files gracefully
        if os.path.exists(img_path):
            # Load as grayscale (H, W)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback for corrupt files
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
        else:
            # Fallback for missing files
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 2. Densified Global Resize
        # Resize to fixed (Width, Height) -> (512, 256)
        # This preserves frequency resolution (Height) and provides dense temporal context (Width)
        img = cv2.resize(
            img, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

        # 3. Channel Replication
        # Convert Grayscale (H, W) -> RGB (H, W, 3)
        # This adapts the 1-channel spectrogram for the 3-channel ResNet backbone
        img = np.stack([img, img, img], axis=-1)

        # 4. Pixel Normalization (0-1)
        img = img.astype(np.float32) / 255.0

        # 5. Convert to Tensor (H, W, C) -> (C, H, W)
        img = torch.tensor(img).permute(2, 0, 1)

        # 6. Augmentation: SpecAugment
        # Applied only in training mode
        if self.mode == "train":
            img = self.apply_spec_augment(img)

        # 7. Standardization (ImageNet Statistics)
        # (Tensor - Mean) / Std
        mean = torch.tensor(Config.NORM_MEAN).view(3, 1, 1)
        std = torch.tensor(Config.NORM_STD).view(3, 1, 1)
        img = (img - mean) / std

        # 8. Target Loading
        rec_id = self.df.iloc[idx]["rec_id"]

        if self.soft_labels is not None:
            # Use distilled soft labels if provided
            target = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        elif self.mode in ["train", "val"]:
            # Use hard ground truth labels
            # Extract binary vector for species_0 ... species_18
            labels = self.df.iloc[idx][self.label_cols].values.astype(np.float32)
            target = torch.tensor(labels, dtype=torch.float32)
        else:
            # Test mode: return dummy zeros
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        return img, target, rec_id

    def apply_spec_augment(self, img):
        """
        Applies SpecAugment (Time and Frequency Masking) to the tensor.
        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W).
        Returns:
            torch.Tensor: Augmented tensor.
        """
        C, H, W = img.shape

        # Hyperparameters for masking (Conservative settings)
        freq_mask_param = 30
        time_mask_param = 40
        num_freq_masks = 2
        num_time_masks = 2

        # Frequency Masking (Masking rows)
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, H - f))
            img[:, f0 : f0 + f, :] = 0.0

        # Time Masking (Masking columns)
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, W - t))
            img[:, :, t0 : t0 + t] = 0.0

        return img


def load_metadata():
    """
    Loads the train, validation, and test metadata CSVs generated by the metadata script.
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    return train_df, val_df, test_df


def get_dataloaders(
    train_df=None,
    val_df=None,
    test_df=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    train_soft_labels=None,
):
    """
    Creates DataLoaders for the provided dataframes.

    Args:
        train_df (pd.DataFrame, optional): Training metadata.
        val_df (pd.DataFrame, optional): Validation metadata.
        test_df (pd.DataFrame, optional): Test metadata.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        train_soft_labels (np.ndarray, optional): Soft labels for the training set (used in distillation).

    Returns:
        tuple: (train_loader, val_loader, test_loader) - None if df is not provided.
    """
    train_loader = None
    val_loader = None
    test_loader = None

    if train_df is not None:
        train_dataset = BirdDataset(
            train_df, mode="train", soft_labels=train_soft_labels
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if val_df is not None:
        val_dataset = BirdDataset(val_df, mode="val")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    if test_df is not None:
        test_dataset = BirdDataset(test_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
