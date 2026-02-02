import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import PathConfig, AudioConfig, DataConfig, TrainConfig


# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(TrainConfig.SEED)


class SpeechCommandsDataset(Dataset):
    def __init__(
        self, dataframe, audio_dir, noise_files=None, is_training=False, transform=None
    ):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filepath' and 'label'.
            audio_dir (str): Root directory for audio files.
            noise_files (list): List of paths to background noise files for silence synthesis.
            is_training (bool): Whether to apply augmentation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_training = is_training
        self.transform = transform

        # Audio parameters
        self.sample_rate = AudioConfig.SAMPLE_RATE
        self.num_samples = AudioConfig.NUM_SAMPLES

        # Preload noise files for dynamic silence synthesis
        self.noise_clips = []
        if noise_files:
            for nf in noise_files:
                try:
                    # Load the full noise file
                    waveform, sr = torchaudio.load(nf)
                    # Resample if necessary
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    self.noise_clips.append(waveform)
                except Exception as e:
                    pass  # Skip corrupted noise files

        # Mel Spectrogram Transform
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            n_mels=AudioConfig.N_MELS,
            f_min=AudioConfig.F_MIN,
            f_max=AudioConfig.F_MAX,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment Transforms
        if self.is_training and TrainConfig.USE_SPECAUGMENT:
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=20
            )

    def __len__(self):
        return len(self.dataframe)

    def _load_audio(self, filepath):
        full_path = os.path.join(self.audio_dir, filepath)
        try:
            waveform, sr = torchaudio.load(full_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        except Exception:
            # Return silent tensor if file load fails
            return torch.zeros(1, self.num_samples)

    def _pad_truncate(self, waveform):
        # Ensure waveform is exactly self.num_samples
        c, t = waveform.shape
        if t > self.num_samples:
            if self.is_training:
                # Random crop for training
                offset = random.randint(0, t - self.num_samples)
                waveform = waveform[:, offset : offset + self.num_samples]
            else:
                # Center crop for validation/test
                offset = (t - self.num_samples) // 2
                waveform = waveform[:, offset : offset + self.num_samples]
        elif t < self.num_samples:
            # Center pad
            padding = self.num_samples - t
            offset = padding // 2
            waveform = torch.nn.functional.pad(waveform, (offset, padding - offset))

        return waveform

    def _get_silence_sample(self):
        if not self.noise_clips:
            return torch.zeros(1, self.num_samples)

        # Pick a random noise file
        noise_wav = random.choice(self.noise_clips)
        c, t = noise_wav.shape

        if t <= self.num_samples:
            repeats = (self.num_samples // t) + 1
            noise_wav = noise_wav.repeat(1, repeats)
            t = noise_wav.shape[1]

        # Random crop
        offset = random.randint(0, t - self.num_samples)
        waveform = noise_wav[:, offset : offset + self.num_samples]

        # Random gain for silence (0.1 to 0.8 to avoid overpowering)
        gain = random.uniform(0.1, 0.8)
        return waveform * gain

    def compute_log_melspec(self, waveform):
        # waveform: (1, samples)
        spec = self.melspec_transform(waveform)
        log_spec = self.db_transform(spec)
        return log_spec

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label_str = row["label"]
        filepath = row["filepath"]

        # 1. Get Waveform
        # If label is silence and we are training (or it's a dummy placeholder), synthesize
        if label_str == "silence":
            if filepath == "dummy" or self.is_training:
                waveform = self._get_silence_sample()
            else:
                # Validation/Test with real silence file -> Deterministic load
                waveform = self._load_audio(filepath)
                waveform = self._pad_truncate(waveform)
        else:
            waveform = self._load_audio(filepath)
            waveform = self._pad_truncate(waveform)

        # 2. Compute Spectrogram -> Shape: (1, n_mels, time)
        log_melspec = self.compute_log_melspec(waveform)

        # 3. Augmentation (SpecAugment)
        if self.is_training and TrainConfig.USE_SPECAUGMENT:
            log_melspec = self.time_masking(log_melspec)
            log_melspec = self.freq_masking(log_melspec)

        # 4. Label Encoding
        label_id = DataConfig.LABEL2ID.get(label_str, DataConfig.LABEL2ID["unknown"])

        return log_melspec, torch.tensor(label_id, dtype=torch.long)


def get_balanced_train_data(load_cached_data=True):
    """
    Loads training metadata, balances the classes, and caches the result.
    """
    cache_path = PathConfig.TRAIN_CACHE

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading balanced training data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing training metadata for balancing...")
    df = pd.read_csv(PathConfig.TRAIN_META)

    # Separate classes
    df_targets = df[df["label"].isin(DataConfig.TARGET_LABELS)]
    df_unknown = df[df["label"] == "unknown"]

    # Downsample 'unknown'
    n_unknown = int(len(df_unknown) * DataConfig.UNKNOWN_SAMPLE_WEIGHT)
    df_unknown_sampled = df_unknown.sample(n=n_unknown, random_state=TrainConfig.SEED)

    # Create 'silence' placeholders
    # We aim for silence to be roughly represented similar to a target class
    avg_target_count = len(df_targets) / 10
    n_silence = int(avg_target_count * 1.5)

    silence_records = [
        {"filepath": "dummy", "label": "silence", "subject_id": "synthetic"}
        for _ in range(n_silence)
    ]
    df_silence_balanced = pd.DataFrame(silence_records)

    # Combine
    df_balanced = pd.concat(
        [df_targets, df_unknown_sampled, df_silence_balanced], ignore_index=True
    )

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=TrainConfig.SEED).reset_index(
        drop=True
    )

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_balanced.to_parquet(cache_path)

    print(f"Balanced training data saved. Total samples: {len(df_balanced)}")

    return df_balanced


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # 1. Prepare DataFrames
    df_train = get_balanced_train_data(load_cached_data=load_cached_data)
    df_val = pd.read_csv(PathConfig.VAL_META)
    df_test = pd.read_csv(PathConfig.TEST_META)

    # Debug mode
    if TrainConfig.DEBUG:
        df_train = df_train.head(100)
        df_val = df_val.head(100)
        df_test = df_test.head(100)

    # 2. Get Noise Files for Silence Synthesis
    # Read original train meta to find background noise files
    df_full_train = pd.read_csv(PathConfig.TRAIN_META)
    noise_files = (
        df_full_train[df_full_train["label"] == "silence"]["filepath"].unique().tolist()
    )
    noise_files = [os.path.join(PathConfig.INPUT_ROOT, f) for f in noise_files]

    # 3. Create Datasets
    train_dataset = SpeechCommandsDataset(
        df_train, PathConfig.INPUT_ROOT, noise_files=noise_files, is_training=True
    )

    val_dataset = SpeechCommandsDataset(
        df_val, PathConfig.INPUT_ROOT, noise_files=noise_files, is_training=False
    )

    test_dataset = SpeechCommandsDataset(
        df_test, PathConfig.INPUT_ROOT, noise_files=None, is_training=False
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
