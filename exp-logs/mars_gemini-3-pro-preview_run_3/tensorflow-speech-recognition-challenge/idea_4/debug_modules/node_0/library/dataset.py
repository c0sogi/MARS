import os
import hashlib
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library import config, augmentations

# Ensure cache directory exists
os.makedirs(config.CACHE_DIR, exist_ok=True)


class MultiResSpectrogram(nn.Module):
    """
    Generates a 3-channel Log-Mel Spectrogram where each channel corresponds to
    a different STFT window size (resolution).
    """

    def __init__(
        self,
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        hop_length=config.HOP_LENGTH,
        window_sizes=config.WINDOW_SIZES,
    ):
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=win,
                    win_length=win,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    center=True,
                    power=2.0,
                )
                for win in window_sizes
            ]
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform):
        # waveform shape: (1, samples) or (samples,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        specs = []
        for t in self.transforms:
            # Output shape: (1, n_mels, time)
            spec = t(waveform)
            specs.append(spec)

        # Stack along channel dimension: (3, n_mels, time)
        multi_res_spec = torch.cat(specs, dim=0)

        # Convert to dB
        multi_res_spec = self.db_transform(multi_res_spec)

        return multi_res_spec


class SpeechDataset(Dataset):
    def __init__(
        self, df, cache_dir=config.CACHE_DIR, is_train=False, load_cached_data=True
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.load_cached_data = load_cached_data

        # Feature extractor
        self.feature_extractor = MultiResSpectrogram()

        # Augmentation
        self.augment = augmentations.SpecAugment() if is_train else None

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath, offset=0, duration=None):
        """Loads audio and pads/crops to config.AUDIO_LEN."""
        full_path = os.path.join(config.INPUT_ROOT, filepath)

        # Determine number of frames to read
        frames_to_read = config.AUDIO_LEN
        if duration is not None:
            frames_to_read = int(duration * config.SAMPLE_RATE)

        start_frame = int(offset * config.SAMPLE_RATE)

        try:
            # Check file info to avoid reading past EOF
            info = sf.info(full_path)
            file_len = info.frames

            # Adjust start_frame and frames_to_read
            if start_frame + frames_to_read > file_len:
                # If we are reading a specific segment (like for silence), we might need to loop or pad.
                # For standard files, we just read what's available.
                pass

            # Read audio
            # sf.read returns (samples, channels) or (samples,)
            waveform, sr = sf.read(
                full_path, start=start_frame, frames=frames_to_read, dtype="float32"
            )

            # Ensure mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Pad or Crop to exactly AUDIO_LEN
            if len(waveform) < config.AUDIO_LEN:
                padding = config.AUDIO_LEN - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode="constant")
            elif len(waveform) > config.AUDIO_LEN:
                waveform = waveform[: config.AUDIO_LEN]

            return torch.from_numpy(waveform).float()

        except Exception as e:
            # Fallback for corrupted files
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(config.AUDIO_LEN).float()

    def _get_silence_sample(self, filepath):
        """Extracts a random 1-second clip from a background noise file."""
        full_path = os.path.join(config.INPUT_ROOT, filepath)
        try:
            info = sf.info(full_path)
            file_len = info.frames
            required_len = config.AUDIO_LEN

            if file_len <= required_len:
                return self._load_audio(filepath)

            if self.is_train:
                # Random crop
                max_start = file_len - required_len
                start_frame = np.random.randint(0, max_start)
            else:
                # Center crop for validation
                start_frame = (file_len - required_len) // 2

            waveform, _ = sf.read(
                full_path, start=start_frame, frames=required_len, dtype="float32"
            )

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            return torch.from_numpy(waveform).float()

        except Exception as e:
            return torch.zeros(config.AUDIO_LEN).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"]

        # Determine if we should use cache
        # We do NOT cache silence samples because they are randomly cropped from long files
        use_cache = self.load_cached_data and (label != config.SILENCE_LABEL)

        spec = None

        if use_cache:
            # Generate hash for filename
            file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{file_hash}.npy")

            if os.path.exists(cache_path):
                try:
                    spec_np = np.load(cache_path)
                    spec = torch.from_numpy(spec_np)
                except:
                    pass  # Failed to load, recompute

        if spec is None:
            # Compute from scratch
            if label == config.SILENCE_LABEL:
                waveform = self._get_silence_sample(filepath)
            else:
                waveform = self._load_audio(filepath)

            # Generate Multi-Resolution Spectrogram
            # Waveform: (Samples,) -> (1, Samples) handled in module
            spec = self.feature_extractor(waveform)

            # Save to cache if applicable
            if use_cache:
                np.save(cache_path, spec.numpy())

        # Apply Augmentation (Train only)
        if self.is_train and self.augment is not None:
            spec = self.augment(spec)

        # Instance Normalization
        # Normalize per sample to mean=0, std=1
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # Get Label ID
        label_id = config.LABEL2ID.get(label, config.LABEL2ID[config.UNKNOWN_LABEL])

        return spec, label_id


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=2):
    """
    Prepares DataLoaders with balanced sampling for training.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Balance Training Data
    # Separate classes
    df_silence = train_df[train_df["label"] == config.SILENCE_LABEL]
    df_unknown = train_df[train_df["label"] == config.UNKNOWN_LABEL]
    df_commands = train_df[
        ~train_df["label"].isin([config.SILENCE_LABEL, config.UNKNOWN_LABEL])
    ]

    # Calculate target count (Median of command classes)
    command_counts = df_commands["label"].value_counts()
    target_count = int(command_counts.median())

    # Undersample Unknown
    if len(df_unknown) > target_count:
        df_unknown = df_unknown.sample(n=target_count, random_state=config.SEED)

    # Oversample Silence
    # Silence files are few (background noise), so we duplicate the rows.
    # The Dataset class handles random cropping, so duplicates result in different audio.
    if len(df_silence) > 0:
        # We want approximately target_count silence samples
        # We sample with replacement
        df_silence = df_silence.sample(
            n=target_count, replace=True, random_state=config.SEED
        )

    # Combine
    train_balanced = (
        pd.concat([df_commands, df_unknown, df_silence])
        .sample(frac=1, random_state=config.SEED)
        .reset_index(drop=True)
    )

    print(f"Balanced Train Size: {len(train_balanced)}")
    print(f"  Commands: {len(df_commands)}")
    print(f"  Unknown:  {len(df_unknown)}")
    print(f"  Silence:  {len(df_silence)}")

    # 3. Create Datasets
    train_dataset = SpeechDataset(train_balanced, is_train=True)
    val_dataset = SpeechDataset(val_df, is_train=False)
    test_dataset = SpeechDataset(test_df, is_train=False)

    # 4. Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
