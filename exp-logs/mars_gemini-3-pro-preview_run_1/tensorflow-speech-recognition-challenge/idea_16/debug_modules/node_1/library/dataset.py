import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config, load_or_create_metadata


class SpeechCommandDataset(Dataset):
    def __init__(
        self,
        df,
        label_encoder=None,
        transform=None,
        is_train=True,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'filepath' and 'fine_label'.
            label_encoder (sklearn.preprocessing.LabelEncoder): Encoder for targets.
            transform: Optional transform (not used directly as we implement specific logic).
            is_train (bool): Flag for training mode (enables augmentation/noise injection).
        """
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.is_train = is_train

        # Audio Parameters
        self.sr = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.target_len = int(self.sr * self.duration)

        # Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Cache Background Noise for Silence Generation and Noise Injection
        self.bg_noise_cache = {}
        self.bg_noise_files = []
        self._cache_background_noise()

    def _cache_background_noise(self):
        """
        Loads all background noise files into memory to speed up
        silence generation and noise injection.
        """
        bg_dir = os.path.join(Config.INPUT_ROOT, "train", "audio", "_background_noise_")
        if os.path.exists(bg_dir):
            files = glob.glob(os.path.join(bg_dir, "*.wav"))
            for fpath in files:
                try:
                    wav, sr = torchaudio.load(fpath)
                    if sr != self.sr:
                        resampler = torchaudio.transforms.Resample(sr, self.sr)
                        wav = resampler(wav)

                    # Ensure mono
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)

                    filename = os.path.basename(fpath)
                    self.bg_noise_cache[filename] = wav
                    self.bg_noise_files.append(filename)
                except Exception as e:
                    print(f"Warning: Failed to load noise file {fpath}: {e}")

    def _get_crop(self, wav, target_len, random_crop=True):
        """Crops the waveform to target_len."""
        length = wav.shape[1]
        if length <= target_len:
            return self._pad_waveform(wav, target_len)

        if random_crop:
            start = random.randint(0, length - target_len)
        else:
            start = (length - target_len) // 2

        return wav[:, start : start + target_len]

    def _pad_waveform(self, wav, target_len):
        """Pads the waveform to target_len."""
        length = wav.shape[1]
        if length >= target_len:
            return wav[:, :target_len]

        padding = target_len - length
        if self.is_train:
            offset = random.randint(0, padding)
        else:
            offset = padding // 2
        return F.pad(wav, (offset, padding - offset))

    def _inject_noise(self, wav):
        """Injects background noise into the waveform."""
        if not self.bg_noise_files or random.random() > 0.5:
            return wav

        noise_file = random.choice(self.bg_noise_files)
        noise_wav = self.bg_noise_cache[noise_file]

        # Crop noise to match signal length
        noise_crop = self._get_crop(noise_wav, self.target_len, random_crop=True)

        # Calculate scaling factor for SNR
        snr_db = random.uniform(10, 30)
        signal_rms = wav.pow(2).mean().sqrt()
        noise_rms = noise_crop.pow(2).mean().sqrt()

        if noise_rms > 1e-6:
            scale = signal_rms / (noise_rms * (10 ** (snr_db / 20)))
            wav = wav + scale * noise_crop

        return wav

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # Determine Label
        label_str = row["fine_label"] if "fine_label" in row else "unknown"

        # --- 1. Load Waveform ---
        if label_str == Config.SILENCE_LABEL:
            # Dynamic Silence Synthesis
            # Use the specific file assigned in metadata to respect splits,
            # or random if not found (fallback).
            filename = os.path.basename(filepath)
            if filename in self.bg_noise_cache:
                wav = self.bg_noise_cache[filename]
            elif self.bg_noise_files:
                wav = self.bg_noise_cache[random.choice(self.bg_noise_files)]
            else:
                wav = torch.zeros(1, self.target_len)

            # Crop: Random for train (diversity), Center for val (determinism)
            wav = self._get_crop(wav, self.target_len, random_crop=self.is_train)

        else:
            # Standard Audio Loading
            full_path = os.path.join(Config.INPUT_ROOT, filepath)
            try:
                wav, sr = torchaudio.load(full_path)
                if sr != self.sr:
                    resampler = torchaudio.transforms.Resample(sr, self.sr)
                    wav = resampler(wav)
            except Exception:
                wav = torch.zeros(1, self.target_len)

            # Convert to Mono if needed
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Pad/Crop to 1s
            wav = self._pad_waveform(wav, self.target_len)

            # Noise Injection (Train only, non-silence)
            if self.is_train:
                wav = self._inject_noise(wav)

        # --- 2. Spectrogram ---
        spec = self.mel_transform(wav)
        spec = self.db_transform(spec)

        # --- 3. SpecAugment (Train only) ---
        if self.is_train:
            # Frequency Masking
            if random.random() < 0.5:
                spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)(spec)
            # Time Masking
            if random.random() < 0.5:
                spec = torchaudio.transforms.TimeMasking(time_mask_param=35)(spec)

        # --- 4. Label Encoding ---
        label = 0
        if self.label_encoder:
            try:
                label = self.label_encoder.transform([label_str])[0]
            except ValueError:
                # Fallback for unknown labels if any
                label = 0

        return spec, label


class MixupCollate:
    """
    Collate function that applies Mixup augmentation to a batch.
    Returns: mixed_inputs, targets_a, targets_b, lam
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        # batch: list of (spec, label)
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])

        batch_size = inputs.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]

        return mixed_inputs, targets_a, targets_b, lam
