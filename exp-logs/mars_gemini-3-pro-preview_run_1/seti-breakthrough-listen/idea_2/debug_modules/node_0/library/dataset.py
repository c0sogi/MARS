import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import INPUT_DIR, NUM_WORKERS, IMAGE_HEIGHT, IMAGE_WIDTH, SEED


class TechnosignatureDataset(Dataset):
    """
    Dataset class for loading Technosignature cadence snippets.
    Handles loading .npy files, normalization, and augmentation.
    """

    def __init__(
        self, metadata_path, mode="train", augment=False, debug=False, debug_size=500
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
            debug (bool): If True, limits the dataset size for debugging.
            debug_size (int): Number of samples to use in debug mode.
        """
        self.mode = mode
        self.augment = augment

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Subset for debugging if requested
        if debug:
            df = df.sample(n=min(len(df), debug_size), random_state=SEED).reset_index(
                drop=True
            )

        self.metadata = df

        # Pre-construct full file paths
        # Metadata contains relative paths (e.g., "train/0/id.npy")
        self.file_paths = [
            os.path.join(INPUT_DIR, fp) for fp in self.metadata["file_path"].values
        ]

        # Store targets if they exist (Train/Val)
        if "target" in self.metadata.columns:
            self.targets = self.metadata["target"].values.astype(np.float32)
        else:
            self.targets = None

        # Store IDs (useful for submission generation)
        self.ids = self.metadata["id"].values

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load .npy file
        # Shape: (6, 273, 256) -> (Channels, Time, Frequency)
        try:
            # Load and convert to float32
            data = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for safety (though metadata verification ensures existence)
            data = np.zeros((6, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

        # Instance-wise Normalization
        # Normalize each snippet independently to handle varying noise floors
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-8:
            data = (data - mean) / std
        else:
            data = data - mean

        # Apply Augmentations (Train only)
        # Augmentations must be applied consistently across all 6 channels
        if self.augment:
            # 1. Random Horizontal Flip (Frequency axis = 2)
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=2).copy()

            # 2. Random Vertical Flip (Time axis = 1)
            # Reverses time; valid for linear features like Doppler drifts
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=1).copy()

            # 3. Random Frequency Shift (Translation along axis 2)
            if np.random.rand() < 0.5:
                # Shift up to 1/8th of the width
                max_shift = data.shape[2] // 8
                shift = np.random.randint(-max_shift, max_shift)
                data = np.roll(data, shift, axis=2)

        # Convert to PyTorch Tensor
        # Input shape is (6, 273, 256), which matches (C, H, W) expectation for Conv2d
        data_tensor = torch.from_numpy(data)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return data_tensor, target
        else:
            # For test set, return dummy target
            return data_tensor, torch.tensor(0.0, dtype=torch.float32)


def get_dataloaders(
    train_meta_path,
    val_meta_path,
    test_meta_path,
    batch_size=32,
    debug=False,
    debug_size=500,
):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """

    # Worker init function for reproducibility
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Train Loader
    train_ds = TechnosignatureDataset(
        train_meta_path, mode="train", augment=True, debug=debug, debug_size=debug_size
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Validation Loader
    val_ds = TechnosignatureDataset(
        val_meta_path, mode="val", augment=False, debug=debug, debug_size=debug_size
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    # Test Loader
    test_ds = TechnosignatureDataset(
        test_meta_path, mode="test", augment=False, debug=debug, debug_size=debug_size
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
