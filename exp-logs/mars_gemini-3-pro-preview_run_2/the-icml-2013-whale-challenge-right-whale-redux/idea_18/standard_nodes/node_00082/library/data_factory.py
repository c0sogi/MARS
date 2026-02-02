import os
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config

# Set seed for reproducibility in transforms
torch.manual_seed(Config.SEED)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Call Detection.
    Loads audio files and generates Mel Spectrograms on-the-fly.
    """

    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing file paths and labels.
            phase (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.df = df
        self.phase = phase
        self.input_root = Config.INPUT_ROOT

        # Define Transforms
        # 1. Mel Spectrogram Generation
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            normalized=Config.NORMALIZED,
        )

        # 2. Amplitude to dB (Log Scale) with clamping
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)

        # 3. Augmentation (Frequency Masking)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )

        # 4. Time Masking (Optional, based on Config)
        self.time_masking = None
        if Config.TIME_MASK_PARAM > 0:
            self.time_masking = torchaudio.transforms.TimeMasking(
                time_mask_param=Config.TIME_MASK_PARAM
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.input_root, row["file_path"])

        # Load Audio
        try:
            # waveform shape: [Channels, Time]
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Fallback for read errors: create silent waveform
            # Approx 2 seconds * 2000 Hz = 4000 samples
            waveform = torch.zeros(1, 4000)
            sr = Config.SAMPLE_RATE

        # Resample if sample rate mismatches (safety check)
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=Config.SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Enforce fixed length (2 seconds * 2000 Hz = 4000 samples)
        # Cite debug_lesson_7
        target_length = 4000
        if waveform.shape[-1] < target_length:
            padding = target_length - waveform.shape[-1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[-1] > target_length:
            waveform = waveform[..., :target_length]

        # Generate Mel Spectrogram -> [Channels, n_mels, time]
        spec = self.mel_spectrogram(waveform)

        # Convert to dB
        spec = self.amplitude_to_db(spec)

        # Instance Standardization (Zero-Mean, Unit-Variance per clip)
        if Config.INSTANCE_NORM:
            mean = spec.mean()
            std = spec.std()
            # Add small epsilon to avoid division by zero
            spec = (spec - mean) / (std + 1e-6)

        # Apply Augmentation (Train only)
        if self.phase == "train":
            spec = self.freq_masking(spec)
            if self.time_masking:
                spec = self.time_masking(spec)

        # Ensure Channel Dimension is 1
        # If input was stereo, average channels
        if spec.shape[0] > 1:
            spec = torch.mean(spec, dim=0, keepdim=True)

        # Get Label
        if "label" in row:
            label = torch.tensor(row["label"], dtype=torch.float32)
        else:
            # Placeholder for test set
            label = torch.tensor(0.0, dtype=torch.float32)

        return spec, label


def get_dataloaders(debug=Config.DEBUG):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Subsample for Debugging
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), 100), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 50), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), 50), random_state=Config.SEED
        ).reset_index(drop=True)

    # Initialize Datasets
    train_dataset = WhaleDataset(train_df, phase="train")
    val_dataset = WhaleDataset(val_df, phase="val")
    test_dataset = WhaleDataset(test_df, phase="test")

    # Configure WeightedRandomSampler for Training
    # This balances the batches given the 90/10 class imbalance
    targets = train_df["label"].values
    class_counts = np.bincount(targets.astype(int))

    # Calculate weights: inverse of frequency
    # Use maximum(..., 1) to prevent division by zero if a class is missing in debug mode
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # Map weights to samples
    sample_weights = class_weights[targets.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
