import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from library.config import CFG


class AudioDataset(Dataset):
    """
    Audio Dataset for loading and preprocessing audio files.

    Features:
    - On-the-fly loading using soundfile
    - Resampling to target sample rate (32kHz)
    - Log-Mel Spectrogram computation
    - Random cropping/padding for training (5s)
    - Full length audio for validation/inference
    - Instance-wise standardization
    - SpecAugment for training
    """

    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and label columns.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.mode = mode

        # Audio Configuration
        self.target_sr = CFG.sample_rate
        self.train_duration = CFG.train_duration
        self.target_len_samples = int(self.target_sr * self.train_duration)

        # Transforms
        # 1. Mel Spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            n_mels=CFG.n_mels,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            power=2.0,
        )

        # 2. Log conversion
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # 3. Augmentation (Train only)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=CFG.freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=CFG.time_mask_param
        )

        # 4. Resampler (Optimized for common case 44.1kHz -> 32kHz)
        self.resampler_44k = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=self.target_sr
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(CFG.INPUT_ROOT, row["filepath"])

        # 1. Load Audio
        try:
            # sf.read is generally faster than torchaudio.load for simple reads
            wav_numpy, sr = sf.read(file_path)
            wav_tensor = torch.from_numpy(wav_numpy).float()
        except Exception as e:
            # Fallback for read errors (should not happen with curated metadata)
            print(f"Warning: Error loading {file_path}: {e}")
            sr = self.target_sr
            wav_tensor = torch.zeros(self.target_len_samples)

        # Ensure shape is (Channels, Time)
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # (1, T)
        else:
            # sf.read returns (T, C) for multi-channel, we need (C, T)
            wav_tensor = wav_tensor.transpose(0, 1)

        # Mix to Mono if necessary (Dataset is mostly mono, but for safety)
        if wav_tensor.shape[0] > 1:
            wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

        # 2. Resample
        if sr != self.target_sr:
            if sr == 44100:
                wav_tensor = self.resampler_44k(wav_tensor)
            else:
                # Functional resample for uncommon rates
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.target_sr
                )
                wav_tensor = resampler(wav_tensor)

        # 3. Length Adjustment (Crop/Pad)
        if self.mode == "train":
            wav_tensor = self._crop_or_pad(wav_tensor)
        else:
            # For Val/Test, we keep full length.
            # Note: Batching variable length tensors requires a custom collate_fn
            # or batch_size=1 in the DataLoader.
            pass

        # 4. Compute Spectrogram
        spec = self.mel_transform(wav_tensor)
        spec = self.db_transform(spec)  # Convert to Log-Mel

        # 5. Normalization (Instance-wise Standardization)
        # Robust against varying gains in recordings
        mean = spec.mean()
        std = spec.std()
        if std > 1e-5:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        # 6. Augmentation (Train only)
        if self.mode == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # 7. Targets
        # Extract labels based on the ordered columns in CFG
        targets = row[CFG.target_columns].values.astype(np.float32)
        target_tensor = torch.tensor(targets)

        return spec, target_tensor

    def _crop_or_pad(self, waveform):
        """
        Crops or pads the waveform to the target length (CFG.train_duration).
        """
        _, length = waveform.shape
        if length < self.target_len_samples:
            # Pad with zeros at the end
            pad_amount = self.target_len_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif length > self.target_len_samples:
            # Random Crop
            start = random.randint(0, length - self.target_len_samples)
            waveform = waveform[:, start : start + self.target_len_samples]

        return waveform
