import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library import config


class SpeechCommandDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands.
    Handles audio loading, padding/truncating, spectrogram conversion,
    instance normalization, and augmentation.
    """

    def __init__(self, metadata_csv, mode="train", transform=None, return_fname=False):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
            transform (callable, optional): Optional external transform.
            return_fname (bool): Whether to return the filename (useful for inference).
        """
        self.df = pd.read_csv(metadata_csv)
        self.mode = mode
        self.return_fname = return_fname
        self.transform = transform

        # Audio parameters
        self.sr = config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES

        # Cache for background noise files to avoid repeated disk I/O
        # Each worker will maintain its own cache (small memory footprint ~15MB)
        self.background_noise_cache = {}

        # Feature Extraction: Log-Mel Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # Augmentation (SpecAugment) - initialized but applied only if mode=='train'
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=config.TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=config.FREQ_MASK_PARAM
        )

    def __len__(self):
        return len(self.df)

    def _get_audio(self, row):
        """
        Loads audio waveform.
        For 'silence' class, extracts a random 1-second chunk from background noise.
        """
        full_path = os.path.join(config.INPUT_ROOT, row["file_path"])

        if row["label"] == "silence":
            # Handle background noise with caching
            if full_path not in self.background_noise_cache:
                waveform, sr = torchaudio.load(full_path)
                # Ensure correct sample rate
                if sr != self.sr:
                    resampler = torchaudio.transforms.Resample(sr, self.sr)
                    waveform = resampler(waveform)
                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                self.background_noise_cache[full_path] = waveform

            full_waveform = self.background_noise_cache[full_path]

            # Random crop 1 second
            if full_waveform.shape[1] > self.num_samples:
                max_offset = full_waveform.shape[1] - self.num_samples
                offset = torch.randint(0, max_offset + 1, (1,)).item()
                waveform = full_waveform[:, offset : offset + self.num_samples]
            else:
                waveform = full_waveform

        else:
            # Regular audio file loading
            waveform, sr = torchaudio.load(full_path)

            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def _process_waveform(self, waveform):
        """
        Pads or truncates the waveform to exactly 1 second.
        """
        c, n = waveform.shape

        if n < self.num_samples:
            # Pad with zeros at the end
            padding = self.num_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif n > self.num_samples:
            # Truncate (Center crop for consistency on commands)
            # Note: Silence is already random cropped in _get_audio
            start = (n - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Audio
        waveform = self._get_audio(row)

        # 2. Fix Length (Pad/Truncate)
        waveform = self._process_waveform(waveform)

        # 3. Generate Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 4. Instance Normalization
        # Standardize each sample individually: (x - mean) / std
        # spec shape: (channels, n_mels, time)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        spec = (spec - mean) / (std + 1e-6)

        # 5. Augmentation (Train only)
        if self.mode == "train":
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 6. Prepare Label
        label_str = row["label"]
        label_id = config.LABEL2ID.get(label_str, config.LABEL2ID["unknown"])

        if self.return_fname:
            return spec, label_id, row["fname"]
        else:
            return spec, label_id


def get_dataloaders(
    train_csv=config.TRAIN_METADATA_PATH,
    val_csv=config.VAL_METADATA_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
):
    """
    Creates DataLoaders for training and validation.
    Applies WeightedRandomSampler to the training set to handle class imbalance.
    """
    # --- Train Dataset & Loader ---
    train_dataset = SpeechCommandDataset(train_csv, mode="train")

    # Calculate weights for WeightedRandomSampler
    labels = train_dataset.df["label"].values
    # Map labels to IDs safely
    label_ids = [config.LABEL2ID.get(l, config.LABEL2ID["unknown"]) for l in labels]

    # Compute class counts
    class_counts = np.bincount(label_ids, minlength=config.NUM_CLASSES)
    class_counts[class_counts == 0] = 1  # Prevent division by zero

    # Compute weights: inverse of frequency
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[label_ids]

    # Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Val Dataset & Loader ---
    val_dataset = SpeechCommandDataset(val_csv, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_csv=config.TEST_METADATA_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
):
    """
    Creates DataLoader for the test set.
    Returns filenames along with data for submission generation.
    """
    test_dataset = SpeechCommandDataset(test_csv, mode="test", return_fname=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader
