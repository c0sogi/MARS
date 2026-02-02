import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class CadenceDataset(Dataset):
    def __init__(self, df, input_dir, img_size=Config.IMG_SIZE):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, target, file_path).
            input_dir (str): Root directory for input data.
            img_size (int): Target spatial dimension (square) for resizing.
        """
        self.df = df
        self.input_dir = input_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load .npy file
        # Original shape: (6, 273, 256) -> (Depth, Freq, Time)
        try:
            data = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for missing files (though metadata check passed)
            # Create zeros of expected shape
            data = np.zeros((6, 273, 256), dtype=np.float32)

        # Convert to tensor
        tensor = torch.from_numpy(data)

        # Resize spatial dimensions
        # Input to interpolate needs to be (N, C, H, W)
        # We treat the 6 cadence positions as channels for the interpolation step
        # Shape: (6, 273, 256) -> unsqueeze -> (1, 6, 273, 256)
        tensor = tensor.unsqueeze(0)

        # Interpolate to (1, 6, 256, 256)
        # align_corners=False is standard for image resizing
        tensor = F.interpolate(
            tensor,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        # The 2D CNN expects input shape (C, H, W) where C=6.
        # Remove the batch dim from interpolate (index 0) to get (6, 256, 256)
        image = tensor.squeeze(0)

        target = torch.tensor(row["target"], dtype=torch.float32)

        return image, target


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/test.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, subsets the data for quick debugging.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Handle Debug Mode
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        # We usually want to predict on full test even in debug, or maybe subset it too.
        # For pipeline checking, subsetting test is fine.
        df_test = df_test.sample(
            n=min(len(df_test), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = CadenceDataset(df_train, Config.INPUT_DIR)
    val_dataset = CadenceDataset(df_val, Config.INPUT_DIR)
    test_dataset = CadenceDataset(df_test, Config.INPUT_DIR)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
