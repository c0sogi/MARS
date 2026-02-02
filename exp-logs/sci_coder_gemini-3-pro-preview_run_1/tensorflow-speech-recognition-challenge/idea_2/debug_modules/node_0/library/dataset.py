import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.seed)


class SpeechCommandsDataset(Dataset):
    def __init__(self, df, phase="train", config=Config):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            phase (str): 'train', 'val', or 'test'.
            config (Config): Configuration class.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.config = config
        self.target_length = config.target_length
        self.sample_rate = config.sample_rate

        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentations (SpecAugment)
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=config.time_mask_param
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=config.freq_mask_param
        )

        # Cache background noise files for dynamic silence generation
        self.noise_cache = {}
        if self.phase == "train":
            self._cache_noise_files()

    def _cache_noise_files(self):
        """
        Pre-load background noise files into memory to speed up dynamic cropping.
        """
        silence_files = self.df[self.df["label"] == "silence"]["filepath"].unique()
        for rel_path in silence_files:
            full_path = os.path.join(self.config.input_root, rel_path)
            if os.path.exists(full_path):
                try:
                    waveform, sr = torchaudio.load(full_path)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    # Mix to mono if necessary
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    self.noise_cache[rel_path] = waveform
                except Exception as e:
                    print(f"Warning: Failed to load noise file {rel_path}: {e}")

    def _load_audio(self, filepath, label):
        """
        Load audio file. If label is 'silence' and phase is 'train',
        perform dynamic random cropping from cached noise files.
        """
        # Dynamic Silence Synthesis
        if (
            self.phase == "train"
            and label == "silence"
            and filepath in self.noise_cache
        ):
            waveform = self.noise_cache[filepath]
            if waveform.shape[1] > self.target_length:
                max_start = waveform.shape[1] - self.target_length
                start = random.randint(0, max_start)
                waveform = waveform[:, start : start + self.target_length]
            return waveform

        # Standard Loading
        full_path = os.path.join(self.config.input_root, filepath)
        if not os.path.exists(full_path):
            # Return silent tensor if file missing (safety fallback)
            return torch.zeros(1, self.target_length)

        try:
            waveform, sr = torchaudio.load(full_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform
        except Exception:
            return torch.zeros(1, self.target_length)

    def _process_waveform(self, waveform):
        """
        Pad or truncate waveform to exactly 1 second.
        """
        length = waveform.shape[1]
        if length < self.target_length:
            # Pad with zeros at the end
            padding = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.target_length:
            # Truncate (take first 1s)
            waveform = waveform[:, : self.target_length]
        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        # 1. Load and Process Audio
        waveform = self._load_audio(filepath, label_str)
        waveform = self._process_waveform(waveform)

        # 2. Generate Spectrogram
        # Output shape: (1, n_mels, time_steps)
        spec = self.mel_transform(waveform)
        spec = self.amplitude_to_db(spec)

        # 3. Instance Normalization
        # Normalize per spectrogram to handle volume variations
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 4. Augmentation (Train Only)
        if self.phase == "train":
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        # 5. Encode Label
        label_id = self.config.label2id.get(label_str, self.config.label2id["unknown"])

        return spec, torch.tensor(label_id, dtype=torch.long)


def get_dataloaders(load_cached_data=False, subset_size=None):
    """
    Generate DataLoaders for train, val, and test sets.
    Handles class balancing for the training set.
    """
    Config.setup()

    # Path for cached balanced training dataframe
    train_cache_path = os.path.join(Config.working_dir, "train_balanced.parquet")

    # ---------------------------------------------------------
    # 1. Prepare Training Data (with Balancing)
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(train_cache_path):
        df_train = pd.read_parquet(train_cache_path)
        print(f"Loaded cached balanced training data: {len(df_train)} samples")
    else:
        # Load raw metadata
        if not os.path.exists(Config.train_metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {Config.train_metadata_path}"
            )

        df_raw = pd.read_csv(Config.train_metadata_path)

        # Balancing Parameters
        TARGET_COUNT = 2000

        # Split by category
        df_silence = df_raw[df_raw["label"] == "silence"]
        df_unknown = df_raw[df_raw["label"] == "unknown"]
        df_targets = df_raw[~df_raw["label"].isin(["silence", "unknown"])]

        balanced_dfs = []

        # A. Handle 'unknown' (Downsample)
        if len(df_unknown) > TARGET_COUNT:
            df_unknown = df_unknown.sample(n=TARGET_COUNT, random_state=Config.seed)
        balanced_dfs.append(df_unknown)

        # B. Handle 'silence' (Upsample/Duplicate)
        # We duplicate rows; the Dataset class handles dynamic cropping from the noise files
        if not df_silence.empty:
            n_repeats = int(np.ceil(TARGET_COUNT / len(df_silence)))
            # Replicate and sample to exact target
            df_silence_balanced = pd.concat([df_silence] * n_repeats)
            df_silence_balanced = df_silence_balanced.sample(
                n=TARGET_COUNT, random_state=Config.seed
            )
            balanced_dfs.append(df_silence_balanced)

        # C. Handle Target Commands (Upsample to balance)
        for label in Config.labels:
            if label in ["silence", "unknown"]:
                continue

            df_cls = df_targets[df_targets["label"] == label]
            if len(df_cls) == 0:
                continue

            if len(df_cls) < TARGET_COUNT:
                # Upsample with replacement
                df_cls = df_cls.sample(
                    n=TARGET_COUNT, replace=True, random_state=Config.seed
                )
            else:
                # Downsample (if any class exceeds target significantly, though unlikely)
                df_cls = df_cls.sample(n=TARGET_COUNT, random_state=Config.seed)
            balanced_dfs.append(df_cls)

        # Combine and Shuffle
        df_train = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=Config.seed)
            .reset_index(drop=True)
        )

        # Cache the balanced dataframe
        df_train.to_parquet(train_cache_path)
        print(f"Created and cached balanced training data: {len(df_train)} samples")

    # ---------------------------------------------------------
    # 2. Prepare Validation and Test Data
    # ---------------------------------------------------------
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    # ---------------------------------------------------------
    # 3. Apply Debugging Subset
    # ---------------------------------------------------------
    if subset_size is not None:
        print(f"Subsetting datasets to {subset_size} samples for debugging.")
        df_train = df_train.iloc[:subset_size]
        df_val = df_val.iloc[:subset_size]
        # We generally don't subset test unless specifically requested for pipeline check
        # but to be consistent with 'subset_size' meaning 'run small', we can subset test too.
        df_test = df_test.iloc[:subset_size]

    # ---------------------------------------------------------
    # 4. Create Datasets and Loaders
    # ---------------------------------------------------------
    train_dataset = SpeechCommandsDataset(df_train, phase="train", config=Config)
    val_dataset = SpeechCommandsDataset(df_val, phase="val", config=Config)
    test_dataset = SpeechCommandsDataset(df_test, phase="test", config=Config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
