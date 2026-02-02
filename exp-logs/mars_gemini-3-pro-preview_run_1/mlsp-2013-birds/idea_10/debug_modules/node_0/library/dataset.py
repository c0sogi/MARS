import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles loading spectrograms, resizing to high-fidelity resolution,
    channel replication, and dynamic augmentation.
    """

    def __init__(self, df, mode="train", pseudo_labels=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'rec_id' and 'file_path'.
            mode (str): 'train', 'val', or 'test'. Controls augmentations.
            pseudo_labels (dict, optional): Dictionary mapping rec_id (int) to
                                            probability vector (np.array).
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.pseudo_labels = pseudo_labels

        # Identify label columns
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-compute image paths to avoid overhead in __getitem__
        self.image_paths = []
        for _, row in self.df.iterrows():
            # Convert WAV path to Spectrogram BMP path
            # src: essential_data/src_wavs/filename.wav
            # target: supplemental_data/spectrograms/filename.bmp
            wav_path = row["file_path"]
            basename = os.path.basename(wav_path)
            bmp_name = os.path.splitext(basename)[0] + ".bmp"
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)
            self.image_paths.append(full_path)

        self.transforms = self.get_transforms()

    def get_transforms(self):
        """
        Returns Albumentations transform pipeline based on mode.
        """
        # ImageNet normalization statistics
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if self.mode == "train":
            return A.Compose(
                [
                    # High-Fidelity Resolution Alignment
                    A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                    # SpecAugment Simulation using CoarseDropout
                    # 1. Time Masking (Vertical strips blocked)
                    A.CoarseDropout(
                        max_holes=2,
                        max_height=Config.IMG_HEIGHT,
                        max_width=int(Config.IMG_WIDTH * 0.1),  # Mask up to 10% of time
                        min_holes=0,
                        min_height=Config.IMG_HEIGHT,
                        min_width=int(Config.IMG_WIDTH * 0.02),
                        fill_value=0,
                        p=0.5,
                    ),
                    # 2. Frequency Masking (Horizontal strips blocked)
                    A.CoarseDropout(
                        max_holes=2,
                        max_height=int(
                            Config.IMG_HEIGHT * 0.15
                        ),  # Mask up to 15% of freq
                        max_width=Config.IMG_WIDTH,
                        min_holes=0,
                        min_height=int(Config.IMG_HEIGHT * 0.02),
                        min_width=Config.IMG_WIDTH,
                        fill_value=0,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation/Test: Deterministic resizing and normalization
            return A.Compose(
                [
                    A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])
        img_path = self.image_paths[idx]

        # Purely Dynamic In-Memory Loading
        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for safety (though EDA showed all exist)
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Convert 1-channel grayscale to 3-channel RGB
        # This adapts the single-channel spectrogram for the ResNet backbone
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        augmented = self.transforms(image=image)
        image_tensor = augmented["image"]

        # Determine Labels
        if self.pseudo_labels is not None and rec_id in self.pseudo_labels:
            # Use pseudo-label if available
            label = self.pseudo_labels[rec_id]
        else:
            # Use ground truth from dataframe
            label = row[self.label_cols].values.astype(np.float32)

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(rec_id, dtype=torch.long),
        )


class MixupCollate:
    """
    Collate function that applies Mixup augmentation to a batch.
    """

    def __init__(self, alpha=Config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels, ids = zip(*batch)

        images = torch.stack(images)
        labels = torch.stack(labels)
        ids = torch.stack(ids)

        # Apply Mixup only if alpha > 0 and batch size > 1
        if self.alpha > 0 and images.size(0) > 1:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Shuffle batch indices
            index = torch.randperm(images.size(0))

            # Mix images and labels
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]

            return mixed_images, mixed_labels, ids

        return images, labels, ids


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    pseudo_labels=None,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders for different splits.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        pseudo_labels (dict, optional): Pseudo-labels for semi-supervised learning.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Select Metadata Path
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
        mode = "train"
    elif split == "val":
        path = Config.VAL_METADATA_PATH
        mode = "val"
    elif split == "test":
        path = Config.TEST_METADATA_PATH
        mode = "test"
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load Metadata
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # Debug Subset
    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Initialize Dataset
    dataset = BirdDataset(df, mode=mode, pseudo_labels=pseudo_labels)

    # Configure Collate Function
    # Apply Mixup only during training
    if mode == "train" and Config.MIXUP_ALPHA > 0:
        collate_fn = MixupCollate(alpha=Config.MIXUP_ALPHA)
    else:
        collate_fn = None  # Default collate

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return loader
