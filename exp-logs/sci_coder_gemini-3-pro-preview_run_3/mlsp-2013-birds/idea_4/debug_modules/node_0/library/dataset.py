import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class BirdDataset(Dataset):
    def __init__(self, df, phase="train", load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, labels).
            phase (str): 'train', 'val', or 'test'. Controls augmentations.
            load_cached_data (bool): Whether to load from cache or re-compute.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.load_cached_data = load_cached_data
        self.num_classes = Config.NUM_CLASSES

        # Audio Parameters
        self.target_length = Config.SR * Config.DURATION  # 160000 samples

        # Spectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # Augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=60)

    def __len__(self):
        return len(self.df)

    def get_label_vector(self, label_str):
        """Converts label string (e.g., '0 4') to multi-hot vector."""
        vec = np.zeros(self.num_classes, dtype=np.float32)
        if pd.isna(label_str) or label_str == "?" or str(label_str).strip() == "":
            return vec

        try:
            indices = [int(x) for x in str(label_str).split()]
            for idx in indices:
                if 0 <= idx < self.num_classes:
                    vec[idx] = 1.0
        except ValueError:
            pass
        return vec

    def load_or_generate_spectrogram(self, rec_id, rel_path):
        """
        Loads spectrogram from cache or generates it from raw WAV.
        Strictly follows the caching logic requirement.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                spec = np.load(cache_path)
                return torch.from_numpy(spec)
            except Exception:
                # If load fails, proceed to compute
                pass

        # 2. Compute from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load audio
        try:
            # soundfile reads as (samples, channels) or (samples,)
            audio, sr = sf.read(full_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel if multi-channel

            # Resample if necessary (though dataset is 16k)
            if sr != Config.SR:
                # Simple resampling not implemented to keep dependencies minimal,
                # assuming dataset is consistent as per description.
                # If needed, one would use librosa.resample or torchaudio.transforms.Resample
                pass

        except Exception as e:
            # Return silent audio in case of file error
            audio = np.zeros(self.target_length)

        # Pad or Truncate
        if len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), "constant")
        else:
            audio = audio[: self.target_length]

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Compute Mel Spectrogram
        spec = self.mel_transform(audio_tensor)
        spec = self.db_transform(spec)

        # Save to cache
        np.save(cache_path, spec.numpy())

        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]
        file_path = row["file_path"]
        label_str = row["labels"]

        # 1. Load Base Spectrogram (1, F, T) -> Squeezed to (F, T) usually, but torchaudio keeps dim
        # MelTransform returns (n_mels, time)
        spec = self.load_or_generate_spectrogram(rec_id, file_path)

        # Ensure shape is (F, T)
        if spec.dim() == 3:
            spec = spec.squeeze(0)

        # 2. Augmentations (Train only)
        if self.phase == "train":
            # Random Time Shift (Roll)
            # Roll along time axis (last dimension)
            if np.random.rand() < 0.5:
                shift = np.random.randint(0, spec.shape[-1])
                spec = torch.roll(spec, shifts=shift, dims=-1)

            # SpecAugment
            # Needs (C, F, T) or (F, T). Transforms expect (..., F, T)
            # We treat it as (1, F, T) for transforms then squeeze back if needed
            spec_unsqueezed = spec.unsqueeze(0)
            spec_unsqueezed = self.freq_mask(spec_unsqueezed)
            spec_unsqueezed = self.time_mask(spec_unsqueezed)
            spec = spec_unsqueezed.squeeze(0)

            # Photometric Augmentation (Brightness/Contrast)
            if np.random.rand() < 0.5:
                # Contrast
                factor = np.random.uniform(0.8, 1.2)
                spec = spec * factor
                # Brightness
                bias = np.random.uniform(-5.0, 5.0)  # dB scale
                spec = spec + bias

        # 3. Normalization (Instance-wise Min-Max to [0, 1])
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 1e-6:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = torch.zeros_like(spec)

        # 4. 3-Channel Replication
        # (F, T) -> (3, F, T)
        image = spec.unsqueeze(0).expand(3, -1, -1)

        # 5. Label
        label = self.get_label_vector(label_str)

        return image, torch.tensor(label, dtype=torch.float32), rec_id


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to a batch of data.
    Returns:
        mixed_x: Mixed inputs
        y_a: Targets for first image
        y_b: Targets for second image
        lam: Lambda mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device == "cuda":
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_dataloaders(train_df, val_df, test_df, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    train_dataset = BirdDataset(
        train_df, phase="train", load_cached_data=load_cached_data
    )
    val_dataset = BirdDataset(val_df, phase="val", load_cached_data=load_cached_data)
    test_dataset = BirdDataset(test_df, phase="test", load_cached_data=load_cached_data)

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
