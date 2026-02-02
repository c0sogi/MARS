import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Ensure reproducible behavior
from library.utils import set_seed

set_seed(Config.SEED)


def compute_spectrogram(file_path):
    """
    Loads audio and computes Log-Mel Spectrogram using torchaudio.
    Returns a numpy array of shape (n_mels, time_steps).
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if necessary
        if sample_rate != Config.SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=Config.SR
            )
            waveform = resampler(waveform)

        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Crop to fixed duration
        target_len = Config.SR * Config.DURATION
        current_len = waveform.shape[1]

        if current_len < target_len:
            pad_amount = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif current_len > target_len:
            waveform = waveform[:, :target_len]

        # Compute Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=True,
        )

        spec = mel_transform(waveform)

        # Convert to DB (Log-Mel)
        # top_db=80 restricts the dynamic range to 80dB
        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        spec_db = db_transform(spec)

        # Remove channel dim: (1, n_mels, time) -> (n_mels, time)
        return spec_db.squeeze(0).numpy()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a silent spectrogram in case of error
        return np.zeros(
            (Config.N_MELS, int(Config.SR * Config.DURATION / Config.HOP_LENGTH) + 1),
            dtype=np.float32,
        )


def prepare_data(df, load_cached_data=True):
    """
    Iterates through the dataframe and ensures all spectrograms are cached.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # We don't use tqdm to keep output clean as requested
    for idx, row in df.iterrows():
        rec_id = int(row["rec_id"])
        cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")

        # Check cache
        if load_cached_data and os.path.exists(cache_path):
            continue

        # Compute and Save
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        if os.path.exists(full_path):
            spec = compute_spectrogram(full_path)
            np.save(cache_path, spec)
        else:
            # If file is missing (should be caught by metadata check), create dummy
            dummy_shape = (
                Config.N_MELS,
                int(Config.SR * Config.DURATION / Config.HOP_LENGTH) + 1,
            )
            np.save(cache_path, np.zeros(dummy_shape, dtype=np.float32))


class TimeShifting(A.ImageOnlyTransform):
    """
    Custom Albumentations transform to apply random time shifting (rolling)
    to the spectrogram.
    """

    def __init__(self, always_apply=False, p=0.5):
        super(TimeShifting, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img shape is (H, W, C). Time axis is W (axis 1).
        shift = np.random.randint(0, img.shape[1])
        return np.roll(img, shift, axis=1)


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    """
    # Standard ImageNet normalization
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                TimeShifting(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()])


class BirdDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Load Spectrogram
        cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")
        if os.path.exists(cache_path):
            spec = np.load(cache_path)
        else:
            # Fallback if cache missing (shouldn't happen if prepare_data called)
            spec = np.zeros((Config.N_MELS, 501), dtype=np.float32)

        # Min-Max Normalize per sample to [0, 1]
        # This maps the dynamic range of the clip to the full image range
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 1e-6:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = np.zeros_like(spec)

        # Replicate to 3 channels: (H, W) -> (H, W, 3)
        # Albumentations expects HWC inputs
        img = np.stack([spec, spec, spec], axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Returns (3, H, W) tensor
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # Prepare Labels
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        if self.mode != "test":
            labels_str = str(row["labels"])
            if labels_str != "?" and labels_str.strip():
                try:
                    indices = [int(x) for x in labels_str.split()]
                    # Ensure indices are within bounds
                    indices = [i for i in indices if 0 <= i < self.num_classes]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass
            return img, label_vec
        else:
            # For test set, return rec_id for submission generation
            return img, label_vec, rec_id


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Subset
    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Prepare Cache (Process all unique files needed)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    unique_df = all_df.drop_duplicates(subset=["rec_id"])

    # We always try to load cache first, but compute if missing
    prepare_data(unique_df, load_cached_data=True)

    # Create Datasets
    train_ds = BirdDataset(train_df, mode="train", transform=get_transforms("train"))

    val_ds = BirdDataset(val_df, mode="val", transform=get_transforms("val"))

    test_ds = BirdDataset(test_df, mode="test", transform=get_transforms("test"))

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
