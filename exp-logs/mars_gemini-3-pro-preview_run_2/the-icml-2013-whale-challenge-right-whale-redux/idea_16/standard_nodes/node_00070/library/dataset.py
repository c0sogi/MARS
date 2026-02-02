import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class WhaleDataset(Dataset):
    """
    Dataset class for Right Whale Detection.
    Implements the 'Golden Recipe' for spectrogram generation and caching.
    """

    def __init__(self, mode, load_cached_data=True, transform=None):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load pre-processed data from cache.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform
        self.target_length = Config.SR * 2  # 2 seconds = 4000 samples

        # Select Metadata File
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.metadata = pd.read_csv(self.csv_path)

        # Define deterministic feature extraction transforms
        # These are applied during caching
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            normalized=Config.NORMALIZED_MEL,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=Config.TOP_DB
        )

        # Load data (from cache or process from scratch)
        self.data, self.targets = self._load_data(load_cached_data)

    def _load_data(self, load_cached):
        """
        Handles the caching logic. Loads from .npy if available, else processes audio.
        """
        cache_data_path = os.path.join(Config.CACHE_DIR, f"{self.mode}_data.npy")
        cache_targets_path = os.path.join(Config.CACHE_DIR, f"{self.mode}_targets.npy")

        # 1. Try to load from cache
        if (
            load_cached
            and os.path.exists(cache_data_path)
            and os.path.exists(cache_targets_path)
        ):
            print(
                f"[{self.mode.upper()}] Loading cached data from {Config.CACHE_DIR}..."
            )
            try:
                data = np.load(cache_data_path)
                targets = np.load(cache_targets_path, allow_pickle=True)
                return data, targets
            except Exception as e:
                print(f"[{self.mode.upper()}] Cache load failed ({e}). Reprocessing...")

        # 2. Process from scratch
        print(f"[{self.mode.upper()}] Processing audio files from scratch...")
        data_list = []
        target_list = []

        # Ensure input root is correct
        input_root = Config.INPUT_ROOT

        for idx, row in self.metadata.iterrows():
            file_path = os.path.join(input_root, row["file_path"])

            # Load Audio
            try:
                waveform, sr = torchaudio.load(file_path)
            except Exception as e:
                # Fallback for missing/corrupt files (though metadata check passed)
                waveform = torch.zeros(1, self.target_length)
                sr = Config.SR

            # Resample if necessary
            if sr != Config.SR:
                resampler = torchaudio.transforms.Resample(sr, Config.SR)
                waveform = resampler(waveform)

            # Ensure Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or Crop to fixed length
            current_len = waveform.shape[1]
            if current_len < self.target_length:
                pad_amount = self.target_length - current_len
                # Pad with zeros (silence)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            elif current_len > self.target_length:
                waveform = waveform[:, : self.target_length]

            # Generate Spectrogram
            # Output shape: (1, n_mels, time_steps)
            spec = self.mel_transform(waveform)

            # Convert to dB (Log Scale) with dynamic range clamping
            spec_db = self.db_transform(spec)

            # Instance Standardization (Zero-Mean, Unit-Variance per clip)
            # Critical for convergence with varying noise levels
            mean = spec_db.mean()
            std = spec_db.std()
            spec_norm = (spec_db - mean) / (std + 1e-6)

            data_list.append(spec_norm.numpy())

            # Handle Targets
            if self.mode == "test":
                target_list.append(row["clip"])
            else:
                target_list.append(row["label"])

        # Stack into a single numpy array
        # Shape: (N, 1, F, T)
        data_arr = np.stack(data_list)
        targets_arr = np.array(target_list)

        # 3. Save to cache
        print(f"[{self.mode.upper()}] Saving processed data to cache...")
        np.save(cache_data_path, data_arr)
        np.save(cache_targets_path, targets_arr)

        return data_arr, targets_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load tensor from memory
        # Shape: (1, F, T)
        img = torch.from_numpy(self.data[idx])

        # Apply Augmentation (if any)
        if self.transform:
            img = self.transform(img)

        target = self.targets[idx]

        if self.mode != "test":
            # Return (image, label)
            # Label needs to be float for BCEWithLogitsLoss
            target = torch.tensor(target, dtype=torch.float32)
            return img, target
        else:
            # Return (image, clip_name) for submission
            return img, target


def get_train_transforms():
    """
    Returns SpecAugment transforms for training.
    """
    return torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=10),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
    )


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Includes WeightedRandomSampler for the training set.
    """
    # Initialize Datasets
    # Train gets augmentation
    train_ds = WhaleDataset("train", transform=get_train_transforms())
    # Val and Test get no augmentation
    val_ds = WhaleDataset("val", transform=None)
    test_ds = WhaleDataset("test", transform=None)

    # Create Weighted Sampler for Class Imbalance
    # Extract targets (numpy array of 0s and 1s)
    targets = train_ds.targets.astype(int)
    class_counts = np.bincount(targets)

    # Compute weights: inverse of frequency
    # Add small epsilon to avoid division by zero if a class is missing (unlikely)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Assign weight to each sample
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, test_loader
