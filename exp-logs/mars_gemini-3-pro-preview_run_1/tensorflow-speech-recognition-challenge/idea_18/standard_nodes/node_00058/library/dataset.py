import os
import json
import random
import math
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T

from library.config import Config
from library.utils import set_seed


def load_noise_files(noise_dir, sample_rate):
    """
    Loads all background noise files into memory for fast access during injection.
    """
    noise_clips = []
    if not os.path.exists(noise_dir):
        return noise_clips

    for file in os.listdir(noise_dir):
        if file.endswith(".wav"):
            path = os.path.join(noise_dir, file)
            try:
                waveform, sr = torchaudio.load(path)
                if sr != sample_rate:
                    resampler = T.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                noise_clips.append(waveform)
            except Exception as e:
                print(f"Warning: Failed to load noise file {file}: {e}")
    return noise_clips


class SpeechDataset(Dataset):
    def __init__(self, df, mode="train", class_to_idx=None, noise_clips=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict): Mapping from label string to integer index.
            noise_clips (list): List of noise waveforms for injection.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.class_to_idx = class_to_idx
        self.noise_clips = noise_clips if noise_clips is not None else []

        # Audio Config
        self.sr = Config.SR
        self.duration = Config.DURATION
        self.target_length = int(self.sr * self.duration)

        # Mel Spectrogram Transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )
        self.db_transform = T.AmplitudeToDB(top_db=Config.TOP_DB)

        # SpecAugment Transforms
        self.time_masking = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath, is_silence_class=False):
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        # Load audio
        waveform, sr = torchaudio.load(full_path)

        # Resample if necessary (though dataset is mostly 16k)
        if sr != self.sr:
            resampler = T.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # Handle Silence Class (Random Crop from long file)
        if is_silence_class and waveform.size(1) > self.target_length:
            max_start = waveform.size(1) - self.target_length
            start = random.randint(0, max_start)
            waveform = waveform[:, start : start + self.target_length]

        # Pad or Crop to target length
        if waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.size(1) > self.target_length:
            waveform = waveform[:, : self.target_length]

        return waveform

    def _inject_noise(self, waveform):
        if not self.noise_clips or random.random() > Config.NOISE_PROB:
            return waveform

        noise = random.choice(self.noise_clips)

        # Crop noise to match waveform length
        if noise.size(1) > self.target_length:
            start = random.randint(0, noise.size(1) - self.target_length)
            noise_segment = noise[:, start : start + self.target_length]
        else:
            # Pad if too short (rare for background noise files)
            padding = self.target_length - noise.size(1)
            noise_segment = torch.nn.functional.pad(noise, (0, padding))

        # Calculate RMS
        waveform_rms = waveform.pow(2).mean().sqrt()
        noise_rms = noise_segment.pow(2).mean().sqrt()

        if noise_rms < 1e-6:
            return waveform

        # Select random SNR
        snr_db = random.uniform(Config.NOISE_SNR_MIN, Config.NOISE_SNR_MAX)
        snr = 10 ** (snr_db / 20)

        # Scale noise
        target_noise_rms = waveform_rms / snr
        scaled_noise = noise_segment * (target_noise_rms / noise_rms)

        # Add noise
        augmented = waveform + scaled_noise
        return augmented

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # Determine label
        if self.mode == "test":
            label_idx = -1  # Dummy
            is_silence = False
        else:
            label_str = row["fine_label"]
            label_idx = self.class_to_idx[label_str]
            is_silence = label_str == Config.SILENCE_LABEL

        # 1. Load Waveform
        waveform = self._load_audio(filepath, is_silence_class=is_silence)

        # 2. Waveform Augmentation (Noise Injection)
        # Only inject noise for non-silence training samples
        if self.mode == "train" and not is_silence:
            waveform = self._inject_noise(waveform)

        # 3. Generate Spectrogram
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)

        # 4. SpecAugment
        if self.mode == "train" and random.random() < Config.MASK_PROB:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        return spec, label_idx


def process_metadata(load_cached_data=True):
    """
    Processes metadata to recover fine-grained labels and balance the training set.
    Caches the result to disk.
    """
    cache_train_path = os.path.join(
        Config.CACHE_DIR, "processed_train_metadata.parquet"
    )
    cache_val_path = os.path.join(Config.CACHE_DIR, "processed_val_metadata.parquet")
    cache_test_path = os.path.join(Config.CACHE_DIR, "processed_test_metadata.parquet")
    cache_map_path = os.path.join(
        Config.CACHE_DIR, "metadata_fine.parquet"
    )  # Storing mapping as a small DF or use JSON

    # 1. Try Load Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
            and os.path.exists(cache_map_path)
        ):

            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)

            # Load mapping
            map_df = pd.read_parquet(cache_map_path)
            class_to_idx = dict(zip(map_df["label"], map_df["idx"]))

            return df_train, df_val, df_test, class_to_idx

    # 2. Process from Scratch
    print("Processing metadata from scratch...")

    # Helper to extract fine label from filepath
    def get_fine_label(filepath):
        # filepath format: train/audio/<label>/<file>.wav
        # or train/audio/_background_noise_/<file>.wav
        parts = filepath.split(os.sep)
        folder = parts[-2]
        if folder == "_background_noise_":
            return Config.SILENCE_LABEL
        return folder

    # Load original metadata
    df_train_orig = pd.read_csv(Config.TRAIN_CSV)
    df_val_orig = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Apply label extraction
    df_train_orig["fine_label"] = df_train_orig["filepath"].apply(get_fine_label)
    df_val_orig["fine_label"] = df_val_orig["filepath"].apply(get_fine_label)
    # Test set doesn't have labels, but we add column for consistency
    df_test["fine_label"] = Config.UNKNOWN_LABEL

    # Create Vocabulary (Class to Index)
    # We want all unique labels found in training
    unique_labels = sorted(df_train_orig["fine_label"].unique())
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # 3. Balancing Logic
    # Separate into Targets, Silence, and Aux
    targets = []
    silence = []
    aux = []

    for label in unique_labels:
        subset = df_train_orig[df_train_orig["fine_label"] == label]
        if label in Config.TARGET_LABELS:
            targets.append(subset)
        elif label == Config.SILENCE_LABEL:
            silence.append(subset)
        else:
            aux.append(subset)

    balanced_dfs = []

    # Upsample Targets
    for df_t in targets:
        n_samples = len(df_t)
        if n_samples < Config.TARGET_SAMPLES_PER_CLASS:
            replace = True
            n_needed = Config.TARGET_SAMPLES_PER_CLASS - n_samples
            upsampled = df_t.sample(
                n=n_needed, replace=replace, random_state=Config.SEED
            )
            balanced_dfs.append(pd.concat([df_t, upsampled]))
        else:
            balanced_dfs.append(df_t)

    # Upsample Silence (Treat as Target)
    # Silence usually has very few files, so we repeat them heavily.
    # The Dataset class handles random cropping, so repeats are distinct samples.
    if silence:
        df_s = silence[0]
        n_samples = len(df_s)
        if n_samples < Config.TARGET_SAMPLES_PER_CLASS:
            # We need exactly TARGET_SAMPLES_PER_CLASS total
            # Since n_samples is small (e.g. 4), we sample with replacement
            upsampled = df_s.sample(
                n=Config.TARGET_SAMPLES_PER_CLASS,
                replace=True,
                random_state=Config.SEED,
            )
            balanced_dfs.append(upsampled)
        else:
            balanced_dfs.append(df_s)

    # Keep Aux as is
    for df_a in aux:
        balanced_dfs.append(df_a)

    df_train_balanced = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle
    df_train_balanced = df_train_balanced.sample(
        frac=1, random_state=Config.SEED
    ).reset_index(drop=True)

    # 4. Save Cache
    df_train_balanced.to_parquet(cache_train_path)
    df_val_orig.to_parquet(cache_val_path)
    df_test.to_parquet(cache_test_path)

    # Save mapping
    map_df = pd.DataFrame(list(class_to_idx.items()), columns=["label", "idx"])
    map_df.to_parquet(cache_map_path)

    return df_train_balanced, df_val_orig, df_test, class_to_idx


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading processed metadata from cache.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    set_seed(Config.SEED)

    # 1. Process Metadata
    df_train, df_val, df_test, class_to_idx = process_metadata(
        load_cached_data=load_cached_data
    )

    print(f"Dataset Statistics:")
    print(f"  Train (Balanced): {len(df_train)}")
    print(f"  Val: {len(df_val)}")
    print(f"  Test: {len(df_test)}")
    print(f"  Num Classes: {len(class_to_idx)}")

    # 2. Load Noise Files (for training injection)
    noise_clips = load_noise_files(Config.NOISE_DIR, Config.SR)
    print(f"  Loaded {len(noise_clips)} background noise clips.")

    # 3. Create Datasets
    train_dataset = SpeechDataset(
        df_train, mode="train", class_to_idx=class_to_idx, noise_clips=noise_clips
    )

    val_dataset = SpeechDataset(
        df_val,
        mode="val",
        class_to_idx=class_to_idx,
        noise_clips=None,  # No noise injection in val
    )

    test_dataset = SpeechDataset(
        df_test, mode="test", class_to_idx=None, noise_clips=None  # No labels
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader, class_to_idx
