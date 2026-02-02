import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class LogMelSpectrogram(torch.nn.Module):
    """
    Transform to convert waveform to Log-Mel Spectrogram.
    """

    def __init__(self, sample_rate, n_mels, n_fft, hop_length, fmin, fmax):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, x):
        # x shape: (Batch, Time) or (Channels, Time)
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        return x


class AudioDataset(Dataset):
    """
    Audio Dataset for Tagging.
    Handles loading, cropping, spectrogram generation, and dual-labeling (Hard/Soft).
    """

    def __init__(self, df, mode="train", soft_labels=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            soft_labels (dict, optional): Dictionary mapping fname -> soft_label_vector.
                                          Used for Student training on Noisy data.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.soft_labels = soft_labels

        # Audio Configuration
        self.sr = Config.SR
        self.duration = Config.DURATION
        self.target_length = int(self.sr * self.duration)

        # Identify Label Columns (exclude metadata)
        self.meta_cols = {
            "fname",
            "labels",
            "filepath",
            "label_count",
            "duration",
            "sample_rate",
            "n_channels",
        }
        # In test.csv, columns exist but are 0-initialized.
        self.label_cols = [c for c in df.columns if c not in self.meta_cols]
        self.label_cols.sort()

        # Pre-load hard labels
        if self.label_cols:
            self.hard_labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            # Fallback if no label columns exist
            self.hard_labels = np.zeros(
                (len(self.df), Config.NUM_CLASSES), dtype=np.float32
            )

        # Transforms
        self.spec_extractor = LogMelSpectrogram(
            sample_rate=Config.SR,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            fmin=Config.FMIN,
            fmax=Config.FMAX,
        )

        # Augmentation (Train only)
        self.spec_augment = torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            ),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # 1. Load Audio
        try:
            # sf.read is robust. Returns (samples, channels) or (samples,)
            wav, sr = sf.read(filepath)
            wav = wav.astype(np.float32)
        except Exception as e:
            # Fallback for corrupted files
            wav = np.zeros(self.target_length, dtype=np.float32)
            sr = self.sr

        # Convert to Tensor
        wav_tensor = torch.from_numpy(wav)

        # Handle Stereo -> Mono
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(dim=1)

        # Resample if necessary
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav_tensor = resampler(wav_tensor)

        # 2. Crop or Pad
        current_len = wav_tensor.shape[0]

        if self.mode == "train":
            # Training: Fixed length (Random Crop or Pad)
            if current_len > self.target_length:
                start = np.random.randint(0, current_len - self.target_length)
                wav_tensor = wav_tensor[start : start + self.target_length]
            else:
                padding = self.target_length - current_len
                wav_tensor = torch.nn.functional.pad(wav_tensor, (0, padding))
        else:
            # Val/Test: Full length
            # Note: Variable lengths are handled by collate_fn
            pass

        # 3. Generate Spectrogram
        # Input: (1, Time) -> Output: (1, n_mels, Time)
        spec = self.spec_extractor(wav_tensor.unsqueeze(0))

        # 4. Apply SpecAugment (Train only)
        if self.mode == "train" and Config.SPECAUGMENT:
            spec = self.spec_augment(spec)

        # 5. Get Target (Label)
        # Priority: Soft Labels (if provided and exists for this file) > Hard Labels
        if self.soft_labels is not None and fname in self.soft_labels:
            target = self.soft_labels[fname]
        else:
            target = self.hard_labels[idx]

        return spec, torch.tensor(target, dtype=torch.float32), fname


def collate_fn(batch):
    """
    Custom collate function to handle variable-length spectrograms.
    Pads the time dimension to the maximum length in the batch.
    """
    specs, targets, fnames = zip(*batch)

    # specs are (1, n_mels, time)
    max_time = max([s.shape[2] for s in specs])

    padded_specs = []
    for s in specs:
        pad_amount = max_time - s.shape[2]
        # Pad last dimension (Time)
        # F.pad tuple is (left, right, top, bottom, ...)
        padded = torch.nn.functional.pad(s, (0, pad_amount))
        padded_specs.append(padded)

    specs_tensor = torch.stack(padded_specs)
    targets_tensor = torch.stack(targets)

    return specs_tensor, targets_tensor, fnames


def get_dataloaders(train_df, val_df, test_df, soft_labels=None):
    """
    Factory function to create DataLoaders.

    Args:
        train_df, val_df, test_df: DataFrames for each split.
        soft_labels: Dictionary of soft labels for Student training.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = AudioDataset(train_df, mode="train", soft_labels=soft_labels)
    val_ds = AudioDataset(val_df, mode="val")
    test_ds = AudioDataset(test_df, mode="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation and Test use collate_fn for variable length support
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
