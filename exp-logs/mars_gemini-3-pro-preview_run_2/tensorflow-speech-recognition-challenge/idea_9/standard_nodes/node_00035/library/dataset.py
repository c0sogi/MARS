import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import set_seed

# ==========================================
# Label Definitions
# ==========================================
LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}


class SpeechCommandDataset(Dataset):
    """
    PyTorch Dataset for Speech Command Recognition.
    Handles on-the-fly audio loading, padding, spectrogram generation,
    and augmentations (Noise Injection, SpecAugment).
    """

    def __init__(self, df, split="train", background_noises=None):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.background_noises = background_noises  # Dict: {filename: waveform_tensor}

        # Audio Settings
        self.sample_rate = Config.sample_rate
        self.num_samples = Config.num_samples

        # Mel Spectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.n_fft,
            win_length=Config.win_length,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels,
            f_min=Config.f_min,
            f_max=Config.f_max,
            normalized=True,
        )

        # Amplitude to DB
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # SpecAugment Transforms (Train only)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.freq_mask_param
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.time_mask_param
        )

    def __len__(self):
        return len(self.df)

    def _fix_length(self, sig):
        """Pad or truncate signal to fixed length."""
        sig_len = sig.shape[0]
        if sig_len < self.num_samples:
            # Pad with zeros
            pad_len = self.num_samples - sig_len
            sig = torch.cat([sig, torch.zeros(pad_len)])
        elif sig_len > self.num_samples:
            # Random crop for train, Center crop for val/test
            if self.split == "train":
                start = random.randint(0, sig_len - self.num_samples)
            else:
                start = (sig_len - self.num_samples) // 2
            sig = sig[start : start + self.num_samples]
        return sig

    def _random_crop_noise(self, noise_wav):
        """Extract a random 1s crop from a long noise file."""
        noise_len = noise_wav.shape[0]
        if noise_len <= self.num_samples:
            return self._fix_length(noise_wav)

        start = random.randint(0, noise_len - self.num_samples)
        return noise_wav[start : start + self.num_samples]

    def _mix_noise(self, sig, noise):
        """Mix noise into signal with random SNR."""
        # Calculate RMS
        sig_rms = sig.norm(p=2) / np.sqrt(sig.shape[0])
        noise_rms = noise.norm(p=2) / np.sqrt(noise.shape[0])

        if noise_rms == 0:
            return sig

        # Random SNR
        snr_db = random.uniform(Config.min_snr_db, Config.max_snr_db)
        snr = 10 ** (snr_db / 20)

        # Scale noise
        target_noise_rms = sig_rms / snr
        scaled_noise = noise * (target_noise_rms / (noise_rms + 1e-8))

        return sig + scaled_noise

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = row["label"]
        fname = row["fname"]

        # 1. Load Audio
        if label_str == "silence":
            # For silence class, we use the background noise files directly
            # row['fname'] corresponds to the filename in _background_noise_
            if self.background_noises and fname in self.background_noises:
                full_noise = self.background_noises[fname]
                sig = self._random_crop_noise(full_noise)
            else:
                # Fallback if not cached (should not happen in this design)
                path = os.path.join(Config.input_root, row["file_path"])
                sig, _ = sf.read(path)
                sig = torch.from_numpy(sig).float()
                sig = self._fix_length(sig)
        else:
            # Regular command file
            path = os.path.join(Config.input_root, row["file_path"])
            # sf.read returns numpy array
            try:
                sig, _ = sf.read(path)
                sig = torch.from_numpy(sig).float()
            except Exception:
                # Fallback for corrupt files (return silence)
                sig = torch.zeros(self.num_samples)

            sig = self._fix_length(sig)

            # 2. Noise Injection (Train only)
            if (
                self.split == "train"
                and self.background_noises
                and random.random() < Config.noise_prob
            ):
                # Pick a random noise file
                noise_fname = random.choice(list(self.background_noises.keys()))
                noise_wav = self.background_noises[noise_fname]
                noise_crop = self._random_crop_noise(noise_wav)
                sig = self._mix_noise(sig, noise_crop)

        # 3. Generate Spectrogram
        # Input to mel_transform should be (channel, time) or (time,)
        # torchaudio expects (..., time)
        spec = self.mel_transform(sig)
        spec = self.db_transform(spec)  # (n_mels, time)

        # 4. SpecAugment (Train only)
        if self.split == "train":
            # Expects (channel, freq, time) or (freq, time)
            # Add dummy channel for transform if needed, or just apply
            # FrequencyMasking expects (..., freq, time)
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 5. Normalization (Standardize per sample)
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 6. Final Shape: (1, n_mels, time)
        spec = spec.unsqueeze(0)

        # 7. Target
        target = LABEL2IDX.get(label_str, LABEL2IDX["unknown"])

        return spec, target, fname


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    Handles metadata loading, background noise caching, and sampler creation.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    if debug:
        df_train = df_train.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)

    # 2. Cache Background Noise (for Train/Silence)
    # We load all files in _background_noise_ directory
    background_noises = {}
    if os.path.exists(Config.background_noise_dir):
        for f in os.listdir(Config.background_noise_dir):
            if f.endswith(".wav"):
                path = os.path.join(Config.background_noise_dir, f)
                try:
                    wav, _ = sf.read(path)
                    background_noises[f] = torch.from_numpy(wav).float()
                except Exception as e:
                    print(f"Warning: Failed to load background noise {f}: {e}")

    # 3. Create Datasets
    # Pass background noises only to train dataset for augmentation
    # Also needed for 'silence' class generation in train
    train_dataset = SpeechCommandDataset(
        df_train, split="train", background_noises=background_noises
    )
    val_dataset = SpeechCommandDataset(df_val, split="val", background_noises=None)
    test_dataset = SpeechCommandDataset(df_test, split="test", background_noises=None)

    # 4. Create WeightedRandomSampler for Train
    # Calculate weights
    label_counts = df_train["label"].value_counts()

    # Handle case where silence has very few rows but we want to sample it more
    # The 'silence' rows in df_train correspond to the long files.
    # We want the model to see 'silence' as often as other classes.
    # We assign weights based on the target class distribution we want (uniform).

    weights = []
    num_samples = len(df_train)

    # Weight per class = Total / (NumClasses * ClassCount)
    # This makes each class have equal probability in the batch
    class_weights = {}
    unique_labels = df_train["label"].unique()
    for label in unique_labels:
        count = label_counts[label]
        if count > 0:
            class_weights[label] = num_samples / (len(unique_labels) * count)
        else:
            class_weights[label] = 0

    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in df_train["label"]]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        sampler=sampler,
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
