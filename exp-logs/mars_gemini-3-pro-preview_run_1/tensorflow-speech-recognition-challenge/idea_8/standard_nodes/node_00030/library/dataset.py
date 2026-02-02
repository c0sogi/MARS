import os
import random
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

from library.config import AudioConfig, TrainConfig, LabelConfig
from library.utils import LabelMapper


class SpeechCommandsDataset(Dataset):
    def __init__(self, dataframe, phase="train", noise_files=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filepath' and 'label'.
            phase (str): 'train', 'val', or 'test'.
            noise_files (list): List of loaded noise waveforms (tensors) for augmentation/synthesis.
        """
        self.df = dataframe.reset_index(drop=True)
        self.phase = phase
        self.input_root = "./input"

        # Audio Configuration
        self.sr = AudioConfig.sample_rate
        self.duration = AudioConfig.duration
        self.target_length = int(self.sr * self.duration)

        # Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=AudioConfig.n_fft,
            hop_length=AudioConfig.hop_length,
            n_mels=AudioConfig.n_mels,
            f_min=AudioConfig.fmin,
            f_max=AudioConfig.fmax,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentation Transforms (SpecAugment)
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=AudioConfig.time_mask_param
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=AudioConfig.freq_mask_param
        )

        # Label Mapping
        self.mapper = LabelMapper()

        # Background Noise Bank (Shared across dataset instances if provided)
        self.noise_bank = noise_files if noise_files else []

    def _load_audio(self, filepath):
        """Loads audio and pads/truncates to target length."""
        full_path = os.path.join(self.input_root, filepath)
        waveform, sr = torchaudio.load(full_path)

        # Resample if necessary (though dataset is mostly 16k)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return self._adjust_length(waveform)

    def _adjust_length(self, waveform):
        """Pads or truncates waveform to exactly 1 second."""
        _, length = waveform.shape
        if length < self.target_length:
            padding = self.target_length - length
            # Pad with zeros
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.target_length:
            # Truncate
            waveform = waveform[:, : self.target_length]
        return waveform

    def _get_random_noise_segment(self):
        """Selects a random 1s segment from the noise bank."""
        if not self.noise_bank:
            return torch.zeros(1, self.target_length)

        noise = random.choice(self.noise_bank)
        _, noise_len = noise.shape

        if noise_len <= self.target_length:
            return self._adjust_length(noise)

        start = random.randint(0, noise_len - self.target_length)
        return noise[:, start : start + self.target_length]

    def _add_noise(self, waveform):
        """Injects background noise at a random SNR."""
        noise = self._get_random_noise_segment()

        # Calculate signal power
        signal_rms = waveform.pow(2).mean().sqrt()
        noise_rms = noise.pow(2).mean().sqrt()

        if noise_rms < 1e-6:
            return waveform

        # Select random SNR
        snr_db = random.uniform(AudioConfig.noise_snr_min, AudioConfig.noise_snr_max)
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))

        # Scale noise
        scaled_noise = noise * (target_noise_rms / noise_rms)

        # Mix
        mixed = waveform + scaled_noise
        return torch.clamp(mixed, -1.0, 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # Extract fine-grained label from filepath if not explicitly provided in a clean way
        # Logic: train/audio/<label>/<file>.wav
        # We assume the dataframe has a 'fine_label' column processed during loading
        label_str = row["fine_label"]

        # 1. Load Waveform
        if label_str == "silence":
            # Synthesize silence from noise bank
            waveform = self._get_random_noise_segment()
        else:
            # Load actual speech file
            try:
                waveform = self._load_audio(filepath)
            except Exception:
                # Fallback for corrupt files: return silence
                waveform = torch.zeros(1, self.target_length)

        # 2. Waveform Augmentation (Train only)
        if self.phase == "train" and label_str != "silence":
            # Apply noise injection with 80% probability
            if random.random() < 0.8:
                waveform = self._add_noise(waveform)

        # 3. Generate Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 4. SpecAugment (Train only)
        if self.phase == "train":
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 5. Encode Label
        # For test set, label is dummy
        if self.phase == "test":
            label_idx = -1  # Dummy
        else:
            label_idx = self.mapper.label2idx.get(label_str, 0)  # Default to 0 if issue

        return spec, torch.tensor(label_idx, dtype=torch.long)


def _load_noise_files():
    """Loads all background noise files into memory."""
    noise_dir = os.path.join("./input", "train", "audio", "_background_noise_")
    noise_files = []
    if os.path.exists(noise_dir):
        files = glob.glob(os.path.join(noise_dir, "*.wav"))
        for f in files:
            try:
                wav, sr = torchaudio.load(f)
                if sr != AudioConfig.sample_rate:
                    wav = torchaudio.transforms.Resample(sr, AudioConfig.sample_rate)(
                        wav
                    )
                # Convert to mono
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                noise_files.append(wav)
            except Exception:
                continue
    return noise_files


def _extract_fine_label(filepath):
    """
    Parses the fine-grained label from the filepath.
    Example: 'train/audio/bed/001.wav' -> 'bed'
    Example: 'train/audio/_background_noise_/noise.wav' -> 'silence'
    """
    parts = filepath.split("/")
    # Usually parts are [train, audio, label, filename] or similar relative path
    if len(parts) >= 2:
        label = parts[-2]
        if label == "_background_noise_":
            return "silence"
        return label
    return "unknown"


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders with Variance-Aware Balancing and Caching.
    """
    # Ensure working directory exists
    os.makedirs(TrainConfig.work_dir, exist_ok=True)
    cache_path = os.path.join(TrainConfig.work_dir, "train_balanced.parquet")

    # 1. Load Metadata
    df_train_meta = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # 2. Recover Fine-Grained Labels
    # The metadata 'label' column has 'unknown' for aux classes. We need the real folder name.
    df_train_meta["fine_label"] = df_train_meta["filepath"].apply(_extract_fine_label)
    df_val["fine_label"] = df_val["filepath"].apply(_extract_fine_label)
    # Test labels are unknown/dummy
    df_test["fine_label"] = "unknown"

    # 3. Balancing Logic (with Cache)
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached balanced training data from {cache_path}")
        df_train_balanced = pd.read_parquet(cache_path)
    else:
        print("Balancing training data (Variance-Aware)...")

        # Separate Targets and Aux
        # Target labels include the 10 commands + silence
        target_labels = LabelConfig.target_labels

        df_targets = df_train_meta[df_train_meta["fine_label"].isin(target_labels)]
        df_aux = df_train_meta[~df_train_meta["fine_label"].isin(target_labels)]

        balanced_dfs = []

        # Upsample Targets
        for label in target_labels:
            df_class = df_targets[df_targets["fine_label"] == label]
            if df_class.empty:
                continue

            # Upsample to target_samples (2000)
            n_samples = len(df_class)
            if n_samples < TrainConfig.target_samples:
                df_resampled = resample(
                    df_class,
                    replace=True,
                    n_samples=TrainConfig.target_samples,
                    random_state=TrainConfig.seed,
                )
                balanced_dfs.append(df_resampled)
            else:
                # If already more than target (unlikely for this dataset except 'unknown'), keep as is or downsample?
                # Dataset stats say ~1700 for targets. So we upsample.
                # If we had > 2000, we could downsample, but here we just take all or upsample.
                balanced_dfs.append(df_class)

        # Keep Aux at natural counts
        balanced_dfs.append(df_aux)

        # Combine and Shuffle
        df_train_balanced = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=TrainConfig.seed)
            .reset_index(drop=True)
        )

        # Cache
        print(f"Saving balanced training data to {cache_path}")
        df_train_balanced.to_parquet(cache_path)

    print(f"Balanced Train Size: {len(df_train_balanced)}")
    print(f"Val Size: {len(df_val)}")

    # 4. Load Noise Bank
    noise_files = _load_noise_files()

    # 5. Create Datasets
    train_dataset = SpeechCommandsDataset(
        df_train_balanced, phase="train", noise_files=noise_files
    )
    val_dataset = SpeechCommandsDataset(df_val, phase="val", noise_files=noise_files)
    test_dataset = SpeechCommandsDataset(df_test, phase="test", noise_files=None)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
