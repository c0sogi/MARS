import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import log_message


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification using pre-computed BMP spectrograms.
    """

    def __init__(self, metadata_df, phase="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'rec_id', 'file_path', and 'labels'.
            phase (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.phase = phase
        self.num_classes = Config.NUM_CLASSES

        # Augmentations
        # Cite Lesson 00019: Resize to 224x224 (Rigid adherence to pretrained input)
        # Cite Lesson 00007: RandomAffine (Horizontal translation only)
        # Cite Lesson 00009: ColorJitter (Photometric augmentation)

        transforms = []
        transforms.append(
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True)
        )

        if self.phase == "train":
            # Horizontal translation (time shifting)
            transforms.append(T.RandomAffine(degrees=0, translate=(0.2, 0)))
            # Photometric Jitter
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))

        transforms.append(T.ToTensor())

        # Normalization for ImageNet pre-trained models
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Map WAV path to BMP path
        # Original: essential_data/src_wavs/filename.wav
        # Target: supplemental_data/spectrograms/filename.bmp
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path).replace(".wav", ".bmp")
        bmp_path = os.path.join(
            Config.INPUT_DIR, "supplemental_data", "spectrograms", filename
        )

        # Load Image
        try:
            image = Image.open(bmp_path).convert("RGB")
        except Exception as e:
            # Fallback (should not happen based on verification)
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Apply Transforms
        image = self.transform(image)

        # Labels
        label_str = str(row["labels"])
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        if label_str != "?" and label_str != "nan":
            try:
                indices = [int(x) for x in label_str.split()]
                label_vec[indices] = 1.0
            except ValueError:
                pass

        return image, label_vec, row["rec_id"]


def get_dataloaders(fold_idx=0, load_cached_data=False):
    """
    Returns DataLoaders for the fixed Train/Val split.
    Ignores fold_idx and load_cached_data (kept for compatibility).
    """
    # Load fixed splits (Cite Lesson 00017: Use Iterative Stratification provided in metadata)
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Create Datasets
    train_dataset = BirdDataset(train_df, phase="train")
    val_dataset = BirdDataset(val_df, phase="val")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=False):
    """
    Returns DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA)
    test_dataset = BirdDataset(test_df, phase="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader
