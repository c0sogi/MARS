import os
import math
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import InChiTokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure recognition.
    Implements Fixed-Height preprocessing strategy.
    """

    def __init__(self, df, tokenizer, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (image_id, file_path, InChI).
            tokenizer (InChiTokenizer): Tokenizer instance for label conversion.
            transform (A.Compose): Albumentations transforms.
            is_test (bool): Whether this is the test set (returns image_id instead of label).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

        # Precompute full file paths
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            self.labels = df["InChI"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for broken/missing images: Black image
            image = np.zeros(
                (Config.IMAGE_HEIGHT, Config.IMAGE_HEIGHT, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # Fixed-Height Preprocessing
        # ---------------------------------------------------------
        h, w, c = image.shape
        target_h = Config.IMAGE_HEIGHT

        # Calculate new width maintaining aspect ratio
        if h > 0:
            scale = target_h / h
            new_w = int(w * scale)
        else:
            new_w = target_h

        # Cap width to avoid OOM on extremely wide molecules
        if new_w > Config.MAX_WIDTH:
            new_w = Config.MAX_WIDTH

        # Resize image
        image = cv2.resize(image, (new_w, target_h))

        # Apply transforms (Augmentation + Normalization + ToTensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual fallback
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            image = torch.from_numpy(image)

        # ---------------------------------------------------------
        # Return Logic
        # ---------------------------------------------------------
        if self.is_test:
            image_id = self.df.iloc[idx]["image_id"]
            return image, image_id
        else:
            text = self.labels[idx]
            # Convert text to sequence with SOS and EOS tokens
            seq = self.tokenizer.text_to_sequence(text, add_sos=True, add_eos=True)
            seq_len = len(seq)
            return image, torch.tensor(seq, dtype=torch.long), seq_len


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    mean = Config.MEAN
    std = Config.STD

    if phase == "train":
        return A.Compose(
            [
                # Training augmentations could be added here (e.g., noise, blur)
                # Avoiding geometric transforms that break aspect ratio assumptions
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def collate_fn(batch):
    """
    Custom collate function to handle variable-width images and variable-length sequences.
    Pads images to the max width in the batch (aligned to stride).
    Pads sequences to the max length in the batch.
    """
    # Determine if this is a test batch (tuple of 2) or train batch (tuple of 3)
    is_test = len(batch[0]) == 2

    images = [item[0] for item in batch]

    # ---------------------------------------------------------
    # 1. Pad Images
    # ---------------------------------------------------------
    # Find max width in this batch
    max_w = max([img.shape[2] for img in images])  # img is (C, H, W)

    # Align to stride (e.g., 32 for ResNet)
    stride = 32
    padded_w = math.ceil(max_w / stride) * stride

    batch_size = len(images)
    c, h = images[0].shape[:2]

    # Create padded batch tensor (filled with 0)
    padded_images = torch.zeros(batch_size, c, h, padded_w, dtype=torch.float32)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img

    # ---------------------------------------------------------
    # 2. Return Batch
    # ---------------------------------------------------------
    if is_test:
        image_ids = [item[1] for item in batch]
        return padded_images, image_ids
    else:
        sequences = [item[1] for item in batch]
        lengths = [item[2] for item in batch]

        # Pad sequences
        max_seq_len = max(lengths)
        pad_idx = Config.PAD_IDX

        padded_seqs = torch.full((batch_size, max_seq_len), pad_idx, dtype=torch.long)

        for i, seq in enumerate(sequences):
            l = len(seq)
            padded_seqs[i, :l] = seq

        lengths = torch.tensor(lengths, dtype=torch.long)

        return padded_images, padded_seqs, lengths


def get_dataloaders(debug=False):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsamples datasets for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # Initialize Tokenizer (loads from cache if available)
    tokenizer = InChiTokenizer(load_cached_data=True)

    # Load Metadata DataFrames
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Debug Subsampling
    if debug:
        print(f"Debug Mode: Subsampling data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Instantiate Datasets
    train_dataset = InChiDataset(
        train_df, tokenizer, transform=get_transforms("train"), is_test=False
    )
    val_dataset = InChiDataset(
        val_df, tokenizer, transform=get_transforms("val"), is_test=False
    )
    test_dataset = InChiDataset(
        test_df, tokenizer, transform=get_transforms("test"), is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
