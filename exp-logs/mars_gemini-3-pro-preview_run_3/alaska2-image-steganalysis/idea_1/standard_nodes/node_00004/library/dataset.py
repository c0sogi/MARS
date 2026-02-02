import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_ROOT,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import seed_everything


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the specified mode.

    Args:
        mode (str): "train", "val", or "test".
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class StegoDataset(Dataset):
    """
    Custom Dataset for Steganography Detection.
    Loads images, extracts the Y-channel (Luminance), and applies transformations.
    Supports balanced sampling (Cite solution_lesson_node_00002) to handle 1:N class imbalance.
    """

    def __init__(self, df, transform=None, balanced=False):
        self.df = df
        self.transform = transform
        self.balanced = balanced

        if self.balanced:
            # Group data by image_id to facilitate pair selection
            # We convert to a dictionary of lists for fast access: {image_id: [records]}
            self.image_ids = self.df["image_id"].unique()
            records = self.df.to_dict("records")
            self.grouped_records = {}
            for r in records:
                iid = r["image_id"]
                if iid not in self.grouped_records:
                    self.grouped_records[iid] = []
                self.grouped_records[iid].append(r)

    def __len__(self):
        # If balanced, we produce 2 samples (1 Cover, 1 Stego) per unique image ID
        if self.balanced:
            return 2 * len(self.image_ids)
        return len(self.df)

    def __getitem__(self, idx):
        if self.balanced:
            # Map linear index to group and class
            group_idx = idx // 2
            is_stego = idx % 2

            img_id = self.image_ids[group_idx]
            records = self.grouped_records[img_id]

            if is_stego:
                # Select one random Stego variant (Label 1)
                stego_candidates = [r for r in records if r["label"] == 1]
                if stego_candidates:
                    row = random.choice(stego_candidates)
                else:
                    # Fallback if no stego found (should not happen in correct metadata)
                    row = records[0]
            else:
                # Select the Cover image (Label 0)
                cover_candidates = [r for r in records if r["label"] == 0]
                if cover_candidates:
                    row = cover_candidates[0]
                else:
                    row = records[0]
        else:
            # Standard sequential access
            row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(INPUT_ROOT, row["image_path"])

        # Read Image using OpenCV (BGR format)
        image = cv2.imread(img_path)

        # Safety check for missing or corrupt images
        if image is None:
            # Return a blank image to prevent crash, though metadata validation should catch this
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Convert BGR to YCrCb and extract Y (Luminance) channel
        # Y is at index 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image[:, :, 0:1]  # Keep 3rd dimension: (H, W, 1)

        # Apply Albumentations Transforms
        if self.transform:
            augmented = self.transform(image=y_channel)
            y_channel = augmented["image"]

        # Normalize to [0, 1]
        # ToTensorV2 converts to Tensor but preserves dtype (uint8 -> ByteTensor)
        # We need FloatTensor in range [0, 1]
        if isinstance(y_channel, torch.Tensor):
            y_channel = y_channel.float() / 255.0
        else:
            # Fallback if transform didn't convert to tensor
            y_channel = y_channel.astype(np.float32) / 255.0
            y_channel = torch.from_numpy(y_channel).permute(2, 0, 1)

        # Get Label
        label = torch.tensor(row["label"], dtype=torch.float32)

        return y_channel, label


def get_loaders(
    train_path=TRAIN_METADATA_PATH,
    val_path=VAL_METADATA_PATH,
    test_path=TEST_METADATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    debug=DEBUG,
    debug_sample_size=DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_path (str): Path to train metadata CSV.
        val_path (str): Path to val metadata CSV.
        test_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets the data for quick debugging.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load Metadata DataFrames
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Apply Debug Subsampling
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=SEED
        ).reset_index(drop=True)

    # Initialize Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")  # Test uses the same deterministic transform

    # Initialize Datasets
    train_dataset = StegoDataset(train_df, transform=train_transform)
    val_dataset = StegoDataset(val_df, transform=val_transform)
    test_dataset = StegoDataset(test_df, transform=val_transform)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
