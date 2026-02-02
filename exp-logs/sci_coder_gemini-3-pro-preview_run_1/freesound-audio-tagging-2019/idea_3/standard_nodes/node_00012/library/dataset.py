import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class AudioDataset(Dataset):
    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Debug Mode
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Identify Label Columns
        # Metadata columns: fname, labels, filepath
        # The rest are class labels
        self.meta_cols = ["fname", "labels", "filepath"]
        self.label_cols = [c for c in self.df.columns if c not in self.meta_cols]

        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Optimization: Pre-initialize resampler for common case (44100 -> 32000)
        self.default_resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=Config.SR
        )

        # Augmentations (Train only)
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # 1. Load Audio
        try:
            # sf.read returns (data, samplerate)
            audio, sr = sf.read(filepath)
            audio = torch.from_numpy(audio).float()
        except Exception as e:
            # Fallback for corrupted files (return silence)
            audio = torch.zeros(Config.SR * Config.DURATION)
            sr = Config.SR

        # Convert to Mono if necessary
        if audio.ndim > 1:
            audio = audio.mean(dim=1)

        # 2. Resample if necessary
        if sr != Config.SR:
            if sr == 44100:
                audio = self.default_resampler(audio)
            else:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=Config.SR
                )
                audio = resampler(audio)

        # 3. Length Handling (Crop/Pad)
        # Train: Fixed length (Config.DURATION)
        # Val/Test: Full length (inference on full clip)

        if self.mode == "train":
            target_samples = Config.SR * Config.DURATION
            num_samples = audio.shape[0]

            if num_samples > target_samples:
                # Random Crop
                start = np.random.randint(0, num_samples - target_samples)
                audio = audio[start : start + target_samples]
            elif num_samples < target_samples:
                # Pad with zeros
                pad_amount = target_samples - num_samples
                audio = torch.nn.functional.pad(audio, (0, pad_amount))

        # 4. Compute Spectrogram
        # Input: (time,) -> Output: (n_mels, time)
        spec = self.mel_transform(audio)
        spec = self.db_transform(spec)

        # Add Channel Dimension: (1, n_mels, time)
        spec = spec.unsqueeze(0)

        # 5. SpecAugment (Train only)
        if self.mode == "train":
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        # 6. Get Labels
        # Ensure labels are float32 for BCEWithLogitsLoss
        labels = row[self.label_cols].values.astype(np.float32)
        labels = torch.from_numpy(labels)

        return spec, labels


def collate_fn(batch):
    """
    Collate function to handle variable length spectrograms.
    Pads the time dimension to the maximum length in the batch.
    """
    # batch is a list of tuples (spec, label)
    specs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find max time dimension
    max_time = max([s.shape[-1] for s in specs])

    padded_specs = []
    for s in specs:
        # s shape: (1, n_mels, time)
        pad_amount = max_time - s.shape[-1]
        if pad_amount > 0:
            # Pad last dimension
            s = torch.nn.functional.pad(s, (0, pad_amount))
        padded_specs.append(s)

    # Stack
    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.stack(labels)

    return specs_tensor, labels_tensor


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the batch.
    Returns:
        mixed_x: Mixed inputs
        y_a: Targets for the first image
        y_b: Targets for the second image
        lam: Lambda mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
