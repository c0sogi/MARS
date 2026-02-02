import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from library.config import Config


class SpeechCommandsDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands.
    Handles loading audio, duration normalization, spectrogram computation,
    and class balancing for training.
    """

    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform (augmentation) to be applied.
            load_cached_data (bool): Whether to attempt loading metadata from cache.
        """
        super().__init__()
        self.mode = mode
        self.transform = transform

        # Audio parameters from Config
        self.fs = Config.SAMPLE_RATE
        self.n_samples = Config.N_SAMPLES

        # Ensure working directory exists for cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load and prepare metadata (with caching)
        self.df = self._prepare_metadata(load_cached_data)

        # Initialize Mel Spectrogram Transform
        # We keep it on CPU here; data loader workers usually operate on CPU.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.fs,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
        )

        # SpecAugment (Only for training)
        self.spec_augment = None
        if self.mode == "train":
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=Config.FREQ_MASK_PARAM
                ),
                torchaudio.transforms.TimeMasking(
                    time_mask_param=Config.TIME_MASK_PARAM
                ),
            )

    def _prepare_metadata(self, load_cached_data):
        """
        Loads metadata CSVs.
        For training: performs class balancing (downsampling unknown, upsampling silence).
        Caches the resulting DataFrame to Parquet.
        """
        cache_filename = f"{self.mode}_metadata.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                # If load fails, proceed to create from scratch
                pass

        # 2. Load source CSV
        if self.mode == "train":
            source_path = Config.TRAIN_CSV
        elif self.mode == "val":
            source_path = Config.VAL_CSV
        else:
            source_path = Config.TEST_CSV

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source metadata not found: {source_path}")

        df = pd.read_csv(source_path)

        # 3. Apply Balancing (Only for Train)
        if self.mode == "train":
            # Separate classes
            df_silence = df[df["label"] == "silence"].copy()
            df_unknown = df[df["label"] == "unknown"].copy()
            df_targets = df[~df["label"].isin(["silence", "unknown"])].copy()

            # A. Downsample 'unknown'
            # Use fixed seed for reproducibility
            if len(df_unknown) > Config.UNKNOWN_TRAIN_SAMPLE_COUNT:
                df_unknown = df_unknown.sample(
                    n=Config.UNKNOWN_TRAIN_SAMPLE_COUNT, random_state=Config.SEED
                )

            # B. Upsample 'silence'
            # We want 'silence' to have a comparable count to target classes (approx 2000)
            target_silence_count = 2000
            if not df_silence.empty:
                # Replicate rows. __getitem__ handles random cropping, so duplicates are fine.
                n_repeats = int(np.ceil(target_silence_count / len(df_silence)))
                df_silence = pd.concat([df_silence] * n_repeats, ignore_index=True)
                df_silence = df_silence.iloc[:target_silence_count]

            # Combine and Shuffle
            df = pd.concat([df_targets, df_unknown, df_silence], ignore_index=True)
            df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

        # 4. Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]
        label_str = row["label"]

        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        # 1. Load Audio
        if not os.path.exists(full_path):
            # Fallback for missing files (should be rare given metadata check)
            waveform = torch.zeros(1, self.n_samples)
        else:
            # Load
            waveform, sr = torchaudio.load(full_path)

            # Resample if needed
            if sr != self.fs:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.fs
                )
                waveform = resampler(waveform)

        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Duration Normalization
        # Target length is self.n_samples
        current_len = waveform.shape[1]

        if label_str == "silence":
            # Special handling for background noise: Random Crop
            if current_len > self.n_samples:
                max_start = current_len - self.n_samples
                # Random integer start index
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start_idx : start_idx + self.n_samples]
            else:
                # Pad if too short
                pad_amt = self.n_samples - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        else:
            # Standard handling: Truncate or Pad
            if current_len > self.n_samples:
                # Truncate (take first 1 second)
                waveform = waveform[:, : self.n_samples]
            elif current_len < self.n_samples:
                # Pad with zeros at the end
                pad_amt = self.n_samples - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))

        # 3. Feature Extraction (Log-Mel Spectrogram)
        # waveform shape: [1, n_samples] -> spec shape: [1, n_mels, time]
        spec = self.mel_transform(waveform)

        # Log scale (add epsilon)
        spec = torch.log(spec + 1e-9)

        # Apply SpecAugment if enabled (training mode)
        if self.spec_augment:
            spec = self.spec_augment(spec)

        # 4. Label Processing
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])

        # 5. Optional Transforms
        if self.transform:
            spec = self.transform(spec)

        return spec, torch.tensor(label_id, dtype=torch.long)
