import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class SpeechCommandsDataset(Dataset):
    """
    Custom Dataset for Speech Commands.
    Handles audio loading, dynamic silence generation, spectrogram conversion,
    and SpecAugment.
    """

    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and 'label'.
            phase (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.target_length = int(Config.SAMPLE_RATE * Config.DURATION)

        # Audio Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentation Transforms (SpecAugment)
        # Parameters chosen for 128 mels and ~100 time frames
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath, label):
        """
        Loads audio from disk.
        For 'silence' class, extracts a random 1-second crop from background noise.
        For others, loads the file and pads/crops to 1 second.
        """
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        if label == "silence":
            # Dynamic silence generation
            # Get file info without loading
            info = torchaudio.info(full_path)
            num_frames = info.num_frames

            if num_frames > self.target_length:
                # Random crop
                offset = torch.randint(0, num_frames - self.target_length, (1,)).item()
                waveform, _ = torchaudio.load(
                    full_path, frame_offset=offset, num_frames=self.target_length
                )
            else:
                # Load full and pad later
                waveform, _ = torchaudio.load(full_path)
        else:
            waveform, _ = torchaudio.load(full_path)

        # Ensure correct channels (mix down to mono if necessary)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Adjust Length (Pad or Crop)
        current_len = waveform.shape[1]

        if current_len < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            # Crop
            if self.phase == "train":
                # Random crop for training
                offset = torch.randint(0, current_len - self.target_length, (1,)).item()
            else:
                # Center crop for val/test
                offset = (current_len - self.target_length) // 2
            waveform = waveform[:, offset : offset + self.target_length]

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        # 1. Load Audio
        waveform = self._load_audio(filepath, label_str)

        # 2. Convert to Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 3. Augmentation (Train only)
        if self.phase == "train":
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 4. Instance Normalization
        # Standardize per sample to handle volume variations
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 5. Label Encoding
        # For test set, label might be 'unknown' placeholder, but we still encode it
        # The model output for 'unknown' class will be used.
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])

        return spec, label_id


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements balancing for the training set with caching.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to try loading cached balanced train set.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Metadata
    df_train_raw = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # --- Training Set Balancing & Caching ---
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_balanced.parquet")

    df_train = None

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(train_cache_path):
        try:
            df_train = pd.read_parquet(train_cache_path)
            # Verify columns
            if not {"filepath", "label"}.issubset(df_train.columns):
                df_train = None  # Invalid cache
        except Exception:
            df_train = None  # Corrupt cache

    # 2. Process if needed
    if df_train is None:
        # Balancing Strategy
        # Target: ~2000 samples per class
        TARGET_COUNT = 2000

        balanced_dfs = []

        # Group by label
        groups = df_train_raw.groupby("label")

        for label, group in groups:
            count = len(group)
            if label == "silence":
                # Silence has very few files, upsample heavily (repeat)
                # The Dataset class handles random cropping, so repeats are fine
                resampled = group.sample(
                    n=TARGET_COUNT, replace=True, random_state=Config.SEED
                )
            elif label == "unknown":
                # Downsample unknown
                resampled = group.sample(
                    n=TARGET_COUNT, replace=False, random_state=Config.SEED
                )
            else:
                # Target classes: Upsample if needed, or take all
                # Most targets have ~1700, so we upsample slightly to 2000
                resampled = group.sample(
                    n=TARGET_COUNT, replace=True, random_state=Config.SEED
                )

            balanced_dfs.append(resampled)

        df_train = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

        # Save to cache
        df_train.to_parquet(train_cache_path)

    # --- Debug Mode ---
    if debug:
        df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

    # --- Dataset Creation ---
    train_dataset = SpeechCommandsDataset(df_train, phase="train")
    val_dataset = SpeechCommandsDataset(df_val, phase="val")
    test_dataset = SpeechCommandsDataset(df_test, phase="test")

    # --- DataLoader Creation ---
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
