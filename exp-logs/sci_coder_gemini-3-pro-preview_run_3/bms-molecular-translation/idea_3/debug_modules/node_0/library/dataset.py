import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.tokenizer import Tokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI prediction.
    Reads images from disk and converts InChI labels to token sequences.
    """

    def __init__(self, df, tokenizer, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR
        self.max_len = Config.MAX_TEXT_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Read as grayscale since Config.IN_CHANNELS = 1
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing images (though validation checked this)
            # Create a black image of expected size to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Expand dimensions to (H, W, 1) for Albumentations/Consistency
        image = np.expand_dims(image, axis=2)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback conversion to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 3. Process Label
        if self.is_test:
            return image
        else:
            inchi_text = row["InChI"]
            # Convert text to sequence of integers with padding/truncation
            sequence = self.tokenizer.text_to_sequence(
                inchi_text, add_special_tokens=True, max_length=self.max_len
            )
            return image, torch.tensor(sequence, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Mean and Std for normalization (0.5 maps [0,1] to [-1,1])
    mean = (0.5,)
    std = (0.5,)

    transforms_list = [
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]

    # Note: We avoid geometric augmentations like HorizontalFlip because
    # stereochemistry in molecules (and InChI strings) is sensitive to orientation.

    return A.Compose(transforms_list)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Creates and returns DataLoaders for train, val, and test splits.

    Args:
        batch_size (int): Batch size for training/inference.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 2. Handle Debug Mode
    if debug:
        print(f"Debug mode active: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Initialize Tokenizer
    # We load cached data if available, otherwise rebuild from metadata
    tokenizer = Tokenizer(load_cached_data=True)

    # 4. Create Datasets
    train_dataset = InChiDataset(
        train_df, tokenizer, transform=get_transforms("train"), is_test=False
    )

    val_dataset = InChiDataset(
        val_df, tokenizer, transform=get_transforms("val"), is_test=False
    )

    test_dataset = InChiDataset(
        test_df, tokenizer, transform=get_transforms("test"), is_test=True
    )

    # 5. Create DataLoaders
    # Note: We don't need a custom collate_fn for padding because
    # InChiDataset pads sequences to a fixed Config.MAX_TEXT_LEN.
    # Default collate_fn handles stacking tensors correctly.

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

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader, tokenizer
