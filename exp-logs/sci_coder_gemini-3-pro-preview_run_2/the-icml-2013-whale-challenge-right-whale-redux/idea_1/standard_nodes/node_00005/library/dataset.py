import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import cv2
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Call Detection.
    Loads audio, converts to Log-Mel Spectrogram, resizes, and normalizes.
    """

    def __init__(self, metadata_csv, root_dir, is_test=False, augment=False):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the audio files.
            is_test (bool): If True, returns clip name instead of label.
            augment (bool): If True, applies SpecAugment.
        """
        self.df = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.is_test = is_test
        self.augment = augment

        # Debug mode: use a small subset
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLES]

        # Audio Transforms
        # We initialize them here to avoid recreating them every call
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment Transforms
        if self.augment:
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            )
            self.time_masking = torchaudio.transforms.TimeMasking(
                time_mask_param=Config.TIME_MASK_PARAM
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        rel_path = row["file_path"]
        full_path = os.path.join(self.root_dir, rel_path)

        # 1. Load Audio
        # Initialize with zeros in case of load failure
        target_samples = int(Config.SAMPLE_RATE * Config.DURATION)

        try:
            audio, sr = sf.read(full_path)
            # If multi-channel, average to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            audio = np.zeros(target_samples, dtype=np.float32)

        # 2. Pad or Truncate
        current_samples = len(audio)
        if current_samples < target_samples:
            pad_width = target_samples - current_samples
            audio = np.pad(audio, (0, pad_width), mode="constant")
        else:
            audio = audio[:target_samples]

        # 3. Convert to Tensor and Compute Spectrogram
        waveform = torch.from_numpy(audio).float()
        # MelSpectrogram expects input shape (..., time)
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # Apply SpecAugment if enabled
        # Cite solution_lesson_node_00003
        if self.augment:
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 4. Resize to Image Size (H, W)
        # Convert to numpy for OpenCV
        spec_np = spec.numpy()
        # cv2.resize expects (width, height)
        spec_resized = cv2.resize(spec_np, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))

        # 5. Normalize (Min-Max to 0-1)
        spec_min = spec_resized.min()
        spec_max = spec_resized.max()
        if spec_max - spec_min > 1e-6:
            spec_norm = (spec_resized - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = spec_resized - spec_min  # effectively zeros

        # 6. Format Output
        # Add channel dimension: (1, H, W)
        image = torch.from_numpy(spec_norm).float().unsqueeze(0)

        if self.is_test:
            clip_name = row["clip"]
            return image, clip_name
        else:
            label = row["label"]
            # Return label as (1,) tensor to match model output shape (Batch, 1)
            return image, torch.tensor([label], dtype=torch.float32)


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, val, and test sets.
    Applies WeightedRandomSampler to the training set to handle class imbalance.
    """
    # Initialize Datasets
    # Apply augmentation only to training set
    train_dataset = WhaleDataset(
        Config.TRAIN_CSV, Config.INPUT_ROOT, is_test=False, augment=True
    )
    val_dataset = WhaleDataset(
        Config.VAL_CSV, Config.INPUT_ROOT, is_test=False, augment=False
    )
    test_dataset = WhaleDataset(
        Config.TEST_CSV, Config.INPUT_ROOT, is_test=True, augment=False
    )

    # Calculate Weights for Imbalance Handling
    # We extract labels directly from the dataframe
    train_labels = train_dataset.df["label"].values

    # Count classes
    class_counts = np.bincount(train_labels)
    # Avoid div by zero (though unlikely given data analysis)
    class_counts = np.maximum(class_counts, 1)

    # Calculate inverse weights
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its label
    sample_weights = class_weights[train_labels]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler handles shuffling implicitly
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
