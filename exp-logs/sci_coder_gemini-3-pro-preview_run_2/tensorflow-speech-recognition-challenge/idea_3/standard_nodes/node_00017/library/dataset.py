import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config

# Ensure reproducible behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class SpeechDataset(Dataset):
    """
    PyTorch Dataset for Speech Command Recognition.
    Handles audio loading, padding/cropping, spectrogram generation, and augmentation.
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform (unused, kept for API compatibility).
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_path)

        # Debugging: Use a small subset if configured
        if Config.DEBUG:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        self.audio_dir = Config.INPUT_ROOT
        self.target_sample_rate = Config.SAMPLE_RATE
        self.num_samples = Config.N_SAMPLES
        self.label2id = Config.LABEL2ID

        # Audio Transforms
        # High-resolution Mel Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # Augmentations (Train only)
        self.aug_time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK
        )
        self.aug_freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK
        )

    def __len__(self):
        return len(self.df)

    def _load_audio(self, rel_path):
        """
        Loads audio file, resamples if necessary, and mixes to mono.
        """
        full_path = os.path.join(self.audio_dir, rel_path)

        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(full_path)
        except Exception as e:
            # Fallback for corrupted files (should not happen with provided data)
            # Return silence
            return torch.zeros(1, self.num_samples)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def _process_waveform(self, waveform, is_background):
        """
        Pads or crops the waveform to the target number of samples.
        Handles background noise by random cropping to generate diverse samples.
        """
        c, t = waveform.shape

        if is_background:
            # For background noise, we randomly crop a 1-second segment
            # This allows us to use the long background files effectively
            if t > self.num_samples:
                if self.mode == "train":
                    start = torch.randint(0, t - self.num_samples, (1,)).item()
                else:
                    # Deterministic crop for validation
                    start = (t - self.num_samples) // 2
                waveform = waveform[:, start : start + self.num_samples]
            else:
                # Pad if too short
                padding = self.num_samples - t
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # For standard commands
            if t > self.num_samples:
                # Center crop to capture the command
                start = (t - self.num_samples) // 2
                waveform = waveform[:, start : start + self.num_samples]
            elif t < self.num_samples:
                # Pad with zeros at the end
                padding = self.num_samples - t
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def compute_spectrogram(self, waveform):
        """
        Generates Log-Mel Spectrogram and applies Instance Normalization.
        """
        # 1. Mel Spectrogram
        spec = self.mel_spectrogram(waveform)

        # 2. Log Scale
        spec = self.amplitude_to_db(spec)

        # 3. Instance Normalization
        # Standardize each spectrogram instance to mean=0, std=1
        # This makes the model robust to volume differences
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Audio
        waveform = self._load_audio(row["file_path"])

        # 2. Process Waveform (Crop/Pad)
        # Check if 'is_background' column exists (it might not in test set)
        is_background = row["is_background"] if "is_background" in row else False
        waveform = self._process_waveform(waveform, is_background)

        # 3. Compute Features
        spec = self.compute_spectrogram(waveform)

        # 4. Augmentation (Train only)
        if self.mode == "train":
            spec = self.aug_time_mask(spec)
            spec = self.aug_freq_mask(spec)

        # 5. Label
        label_str = row["label"]
        label_id = self.label2id.get(label_str, self.label2id["unknown"])

        # Return format: (Input, Label, Filename)
        # Input shape: (1, Freq, Time)
        return spec, label_id, row["fname"]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=False
):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements WeightedRandomSampler for the training set to handle class imbalance.
    """

    # 1. Create Datasets
    train_dataset = SpeechDataset(Config.TRAIN_METADATA, mode="train")
    val_dataset = SpeechDataset(Config.VAL_METADATA, mode="val")
    test_dataset = SpeechDataset(Config.TEST_METADATA, mode="test")

    # 2. Create WeightedRandomSampler for Training
    # Extract targets to compute weights
    # This ensures balanced batches despite the 17:1 imbalance of 'unknown' class
    train_targets = [
        train_dataset.label2id.get(label, train_dataset.label2id["unknown"])
        for label in train_dataset.df["label"]
    ]
    train_targets = np.array(train_targets)

    # Count occurrences of each class
    class_counts = np.bincount(train_targets, minlength=Config.NUM_CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Compute weights: inverse of frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its label
    sample_weights = class_weights[train_targets]

    # Create sampler
    # replacement=True is essential for oversampling minority classes
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
