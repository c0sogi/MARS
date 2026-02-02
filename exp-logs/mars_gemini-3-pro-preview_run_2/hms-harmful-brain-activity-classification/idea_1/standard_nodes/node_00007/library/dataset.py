import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from library.config import Config
from library.signal_processing import EEGProcessor


class EEGDataset(Dataset):
    """
    PyTorch Dataset for loading and processing EEG data.

    Integrates with EEGProcessor to convert raw EEG signals into Log-Mel Spectrogram
    images on-the-fly, with built-in caching for efficiency.
    """

    def __init__(self, data, mode="train", transform=None):
        """
        Args:
            data (pd.DataFrame or str): DataFrame containing metadata OR string
                                        ("train", "val", "test") to load from Config paths.
            mode (str): 'train', 'val', or 'test'. Determines return values and offset logic.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.mode = mode
        self.transform = transform

        # Initialize the signal processor
        self.processor = EEGProcessor()

        # Load metadata based on input type
        if isinstance(data, str):
            if data == "train":
                self.df = pd.read_csv(Config.TRAIN_CSV)
            elif data == "val":
                self.df = pd.read_csv(Config.VAL_CSV)
            elif data == "test":
                self.df = pd.read_csv(Config.TEST_CSV)
            else:
                # Fallback if a direct path is provided
                if os.path.exists(data):
                    self.df = pd.read_csv(data)
                else:
                    raise ValueError(f"Invalid data argument: {data}")
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise TypeError(
                "data must be a pandas DataFrame or a string ('train', 'val', 'test')"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract necessary metadata
        eeg_path = row["eeg_path"]
        eeg_id = str(row["eeg_id"])

        # Determine offset and cache_id based on mode
        if self.mode == "test":
            # Test files are pre-cropped to 50s, so offset is 0
            offset = 0.0
            cache_id = f"{eeg_id}_test"
        else:
            # Train/Val files are full length, use offset
            offset = row["eeg_label_offset_seconds"]
            # Use sub_id to distinguish different crops from the same EEG file
            sub_id = str(row["eeg_sub_id"])

            # Disable caching for training to save disk space (Full dataset ~250GB)
            if self.mode == "train":
                cache_id = None
            else:
                cache_id = f"{eeg_id}_{sub_id}"

        # Load and process the EEG data
        # We set load_cached_data=True to use the caching mechanism in EEGProcessor
        # This ensures we don't re-compute spectrograms for the same sample across epochs
        image_tensor = self.processor.load_and_process(
            eeg_path=eeg_path,
            offset_seconds=offset,
            load_cached_data=True,
            cache_id=cache_id,
        )

        # Apply Augmentations / Transforms
        if self.transform:
            # Convert Tensor (C, H, W) -> Numpy (H, W, C) for Albumentations
            img_np = image_tensor.permute(1, 2, 0).numpy()

            # Apply transform
            # Albumentations returns a dictionary
            augmented = self.transform(image=img_np)
            img_np = augmented["image"]

            # Convert back to Tensor (C, H, W)
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Return data based on mode
        if self.mode == "test":
            return image_tensor
        else:
            # Extract target probabilities
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            return image_tensor, torch.tensor(targets, dtype=torch.float32)
