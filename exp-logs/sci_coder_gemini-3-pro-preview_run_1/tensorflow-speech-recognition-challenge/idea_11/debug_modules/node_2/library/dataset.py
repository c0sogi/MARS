import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


def load_background_noises(noise_dir, sample_rate=16000):
    """
    Loads all .wav files from the background noise directory.
    Returns a list of tensors.
    """
    noises = []
    if not os.path.exists(noise_dir):
        return noises

    for filename in os.listdir(noise_dir):
        if filename.endswith(".wav"):
            path = os.path.join(noise_dir, filename)
            try:
                waveform, sr = torchaudio.load(path)
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                noises.append(waveform)
            except Exception as e:
                print(f"Warning: Failed to load noise file {filename}: {e}")
    return noises


class SpeechDataset(Dataset):
    def __init__(self, df, mode, background_noises=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            background_noises (list): List of noise waveforms.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.background_noises = background_noises if background_noises else []
        self.audio_len = Config.AUDIO_LEN
        self.input_root = Config.INPUT_ROOT

    def __len__(self):
        return len(self.df)

    def _get_random_noise(self):
        if not self.background_noises:
            return torch.zeros(1, self.audio_len)
        noise = random.choice(self.background_noises)
        # Random crop
        if noise.shape[1] > self.audio_len:
            start = random.randint(0, noise.shape[1] - self.audio_len)
            return noise[:, start : start + self.audio_len]
        else:
            # Pad if too short
            return self._fix_length(noise)

    def _fix_length(self, waveform):
        """Pads or crops the waveform to Config.AUDIO_LEN."""
        c, t = waveform.shape
        if t > self.audio_len:
            if self.mode == "train":
                start = random.randint(0, t - self.audio_len)
            else:
                start = (t - self.audio_len) // 2
            waveform = waveform[:, start : start + self.audio_len]
        elif t < self.audio_len:
            padding = self.audio_len - t
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _add_noise(self, waveform):
        """Injects background noise with random SNR (10-30dB)."""
        if not self.background_noises or random.random() < 0.5:
            return waveform

        noise = self._get_random_noise()

        # Calculate signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        if noise_power == 0:
            return waveform

        # Random SNR between 10 and 30 dB
        snr_db = random.uniform(10, 30)
        snr = 10 ** (snr_db / 10)

        # Calculate required noise scale
        scale = torch.sqrt(signal_power / (noise_power * snr))

        return waveform + scale * noise

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # Determine Label and Load Audio
        if self.mode == "test":
            # Test mode: Just load audio, label is dummy
            full_path = os.path.join(self.input_root, filepath)
            waveform, sr = torchaudio.load(full_path)
            label_idx = 0  # Dummy
            fname = row["fname"] if "fname" in row else os.path.basename(filepath)

        else:
            # Train/Val mode
            fine_label = row["fine_label"]

            if fine_label == Config.SILENCE_LABEL:
                waveform = self._get_random_noise()
                label_idx = Config.get_class_index(Config.SILENCE_LABEL)
                fname = f"silence_{idx}.wav"
            else:
                full_path = os.path.join(self.input_root, filepath)
                waveform, sr = torchaudio.load(full_path)

                # Retrieve class index
                # If label is not in Config (unlikely with correct filtering), map to unknown?
                # Config.get_class_index raises ValueError if not found.
                try:
                    label_idx = Config.get_class_index(fine_label)
                except ValueError:
                    # Fallback for safety, though balancing should prevent this
                    # Map to a known auxiliary class or handle gracefully
                    # Here we assume dataset prep ensures validity.
                    # If strictly unknown to our fine-grained list, we might map to an 'unknown' bucket
                    # but our Config doesn't have a generic 'unknown' index for training.
                    # We assume the fine_label is valid.
                    label_idx = 0

                fname = os.path.basename(filepath)

        # Fix length
        waveform = self._fix_length(waveform)

        # Augmentation (Train only, and not for silence class which is already noise)
        if self.mode == "train":
            # We don't add noise to silence class as it IS noise
            if self.mode != "test" and row.get("fine_label") != Config.SILENCE_LABEL:
                waveform = self._add_noise(waveform)

        # Squeeze channel dimension: (1, Time) -> (Time,)
        # This ensures DataLoader collates to (Batch, Time)
        waveform = waveform.squeeze(0)

        return waveform, label_idx, fname


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for Train, Val, and Test.
    Implements Variance-Aware Balancing and Caching.
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Background Noises
    noise_dir = os.path.join(Config.TRAIN_AUDIO_DIR, "_background_noise_")
    background_noises = load_background_noises(noise_dir, Config.SAMPLE_RATE)

    # 2. Prepare Training Data
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_balanced.parquet")

    if load_cached_data and os.path.exists(train_cache_path):
        print(f"Loading cached training data from {train_cache_path}")
        df_train = pd.read_parquet(train_cache_path)
    else:
        print("Processing training data...")
        df_raw = pd.read_csv(Config.TRAIN_CSV)

        # Extract fine-grained label from filepath
        # filepath format: train/audio/<label>/<file>.wav
        # We use os.path.dirname to get .../<label> then split
        df_raw["fine_label"] = df_raw["filepath"].apply(
            lambda x: os.path.basename(os.path.dirname(x))
        )

        # Filter out _background_noise_ files from the main list (handled synthetically)
        df_raw = df_raw[df_raw["fine_label"] != "_background_noise_"].copy()

        # Filter only labels that are in our Config.CLASSES (sanity check)
        # This removes any potential weird folders not in the spec
        valid_classes = set(Config.FINE_GRAINED_CLASSES)
        df_raw = df_raw[df_raw["fine_label"].isin(valid_classes)].copy()

        # Balancing Strategy
        dfs_to_concat = []

        # A. Targets: Upsample to ~2000
        target_df = df_raw[df_raw["fine_label"].isin(Config.TARGET_LABELS)]
        for label, group in target_df.groupby("fine_label"):
            if len(group) < 2000:
                upsampled = group.sample(n=2000, replace=True, random_state=Config.SEED)
                dfs_to_concat.append(upsampled)
            else:
                dfs_to_concat.append(group)

        # B. Auxiliaries: Keep all (Variance preservation)
        aux_df = df_raw[~df_raw["fine_label"].isin(Config.TARGET_LABELS)]
        dfs_to_concat.append(aux_df)

        # C. Silence: Synthesize 2000 samples
        silence_data = {
            "filepath": [None] * 2000,  # Handled by __getitem__
            "label": [Config.SILENCE_LABEL] * 2000,
            "subject_id": ["synthetic"] * 2000,
            "fine_label": [Config.SILENCE_LABEL] * 2000,
        }
        dfs_to_concat.append(pd.DataFrame(silence_data))

        df_train = pd.concat(dfs_to_concat, ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=Config.SEED).reset_index(
            drop=True
        )

        # Save to cache
        df_train.to_parquet(train_cache_path)
        print(f"Saved balanced training data to {train_cache_path}")

    # 3. Prepare Validation Data
    # We add some synthetic silence to validation to track silence accuracy
    df_val = pd.read_csv(Config.VAL_CSV)
    df_val["fine_label"] = df_val["filepath"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )

    # Filter valid classes
    valid_classes = set(Config.FINE_GRAINED_CLASSES)
    df_val = df_val[df_val["fine_label"].isin(valid_classes)].copy()

    # Add synthetic silence for validation (e.g., 200 samples)
    val_silence_data = {
        "filepath": [None] * 200,
        "label": [Config.SILENCE_LABEL] * 200,
        "subject_id": ["synthetic"] * 200,
        "fine_label": [Config.SILENCE_LABEL] * 200,
    }
    df_val = pd.concat([df_val, pd.DataFrame(val_silence_data)], ignore_index=True)

    # 4. Prepare Test Data
    df_test = pd.read_csv(Config.TEST_CSV)
    # Test CSV has 'filepath' and 'label' (unknown). No fine_label needed.
    # We ensure 'fname' column exists for submission mapping
    df_test["fname"] = df_test["filepath"].apply(os.path.basename)

    # 5. Create Datasets
    train_dataset = SpeechDataset(
        df_train, mode="train", background_noises=background_noises
    )
    val_dataset = SpeechDataset(df_val, mode="val", background_noises=background_noises)
    test_dataset = SpeechDataset(df_test, mode="test", background_noises=None)

    # 6. Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
