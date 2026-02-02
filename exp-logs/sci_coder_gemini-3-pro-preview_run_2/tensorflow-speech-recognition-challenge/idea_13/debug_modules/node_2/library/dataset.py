import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict

from library.config import PathConfig, AudioConfig, TrainConfig


class SpeechCommandsDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands with High-Fidelity Signal Processing.
    Implements on-the-fly waveform loading, dynamic noise mixing, and spectral oversampling.
    """

    def __init__(
        self,
        split: str = "train",
        transform: bool = True,
        cache_data: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            transform (bool): Whether to apply augmentations (noise mixing, SpecAugment).
            cache_data (bool): Whether to preload waveforms into RAM for speed.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.transform = transform and (split == "train")
        self.cache_data = cache_data
        self.path_config = PathConfig()
        self.audio_config = AudioConfig()
        self.train_config = TrainConfig()

        # Load Metadata
        if split == "train":
            self.metadata_path = self.path_config.train_metadata_path
        elif split == "val":
            self.metadata_path = self.path_config.val_metadata_path
        elif split == "test":
            self.metadata_path = self.path_config.test_metadata_path
        else:
            raise ValueError(f"Invalid split: {split}")

        self.df = pd.read_csv(self.metadata_path)

        # Debug mode: subset data
        if debug:
            print(
                f"[{split}] Debug mode: limiting to {self.train_config.debug_sample_size} samples."
            )
            self.df = self.df.iloc[: self.train_config.debug_sample_size]

        # Label Mapping
        self.label_to_idx = self.audio_config.label_to_idx
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Audio Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_config.sample_rate,
            n_fft=self.audio_config.n_fft,
            win_length=self.audio_config.win_length,
            hop_length=self.audio_config.hop_length,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.fmin,
            f_max=self.audio_config.fmax,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            top_db=self.audio_config.top_db
        )

        # Augmentations
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=self.train_config.time_mask_param
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.train_config.freq_mask_param
        )

        # Cache Setup
        self.waveform_cache: Dict[str, torch.Tensor] = {}
        self.background_noise_files: List[str] = []
        self.background_noise_cache: Dict[str, torch.Tensor] = {}

        # Preload Data
        self._prepare_data()

    def _prepare_data(self):
        """
        Preloads waveforms and identifies background noise files for mixing.
        """
        # 1. Identify Background Noise Files (only relevant for training mixing)
        # In the training set, background noise is also a class ('silence').
        # We also want a separate list of these files to use for mixing into other commands.
        if self.split == "train":
            # Filter from the dataframe where is_background is True
            bg_rows = self.df[self.df["is_background"] == True]
            self.background_noise_files = bg_rows["file_path"].unique().tolist()

            # Preload background noise files completely
            for rel_path in self.background_noise_files:
                full_path = os.path.join(self.path_config.input_dir, rel_path)
                try:
                    wav, sr = torchaudio.load(full_path)
                    # Resample if needed (though EDA says all are 16k)
                    if sr != self.audio_config.sample_rate:
                        resampler = torchaudio.transforms.Resample(
                            sr, self.audio_config.sample_rate
                        )
                        wav = resampler(wav)
                    self.background_noise_cache[rel_path] = wav
                except Exception as e:
                    print(f"Warning: Failed to load background noise {rel_path}: {e}")

        # 2. Preload Main Dataset Waveforms
        if self.cache_data:
            print(f"[{self.split}] Caching {len(self.df)} waveforms to RAM...")
            for idx, row in self.df.iterrows():
                rel_path = row["file_path"]
                # Skip if already cached (e.g. background noise files appearing in df)
                if (
                    rel_path in self.waveform_cache
                    or rel_path in self.background_noise_cache
                ):
                    continue

                full_path = os.path.join(self.path_config.input_dir, rel_path)
                try:
                    wav, sr = torchaudio.load(full_path)
                    if sr != self.audio_config.sample_rate:
                        resampler = torchaudio.transforms.Resample(
                            sr, self.audio_config.sample_rate
                        )
                        wav = resampler(wav)

                    # If it's a command file, we can pad/crop now to save space/time
                    # But if it's a background file used as a sample, we keep it full length
                    # to crop randomly in __getitem__
                    if not row.get("is_background", False):
                        wav = self._pad_or_crop(wav)

                    self.waveform_cache[rel_path] = wav
                except Exception as e:
                    # Create a silent placeholder if file fails
                    print(f"Warning: Failed to load {rel_path}: {e}")
                    self.waveform_cache[rel_path] = torch.zeros(
                        1, self.audio_config.sample_rate * self.audio_config.duration
                    )

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Fixes waveform length to exactly 1 second (sample_rate samples).
        """
        target_len = self.audio_config.sample_rate * self.audio_config.duration
        channels, current_len = waveform.shape

        if current_len == target_len:
            return waveform
        elif current_len > target_len:
            # Center crop
            start = (current_len - target_len) // 2
            return waveform[:, start : start + target_len]
        else:
            # Pad with zeros
            padding = target_len - current_len
            # Pad equally on both sides
            pad_left = padding // 2
            pad_right = padding - pad_left
            return torch.nn.functional.pad(waveform, (pad_left, pad_right))

    def _mix_background_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Mixes a random segment of background noise into the waveform.
        """
        if not self.background_noise_cache:
            return waveform

        # Pick a random noise file
        noise_path = random.choice(self.background_noise_files)
        noise_wav = self.background_noise_cache[noise_path]

        # Pick a random 1s slice
        noise_len = noise_wav.shape[1]
        target_len = self.audio_config.sample_rate * self.audio_config.duration

        if noise_len > target_len:
            start = random.randint(0, noise_len - target_len)
            noise_slice = noise_wav[:, start : start + target_len]
        else:
            noise_slice = self._pad_or_crop(noise_wav)

        # Mix with random SNR (Signal-to-Noise Ratio)
        # Simple implementation: random weight between 0.0 and 0.1 (conservative)
        # Or more robust: random factor
        noise_energy = torch.sum(noise_slice**2)
        if noise_energy < 1e-6:
            return waveform

        signal_energy = torch.sum(waveform**2)
        if signal_energy < 1e-6:
            return waveform  # Signal is silence, don't just add noise if it's supposed to be silence?
            # Actually if label is silence, we don't call this function usually.

        # Random weight for noise [0.0, 0.5]
        noise_weight = random.uniform(0.0, 0.3)
        return waveform + noise_weight * noise_slice

    def _get_waveform(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves the waveform for the given index.
        Handles random cropping for 'silence' class (background noise files).
        """
        row = self.df.iloc[idx]
        rel_path = row["file_path"]
        is_background = row.get("is_background", False)

        # Retrieve from cache or load
        if is_background and rel_path in self.background_noise_cache:
            raw_wav = self.background_noise_cache[rel_path]
        elif rel_path in self.waveform_cache:
            raw_wav = self.waveform_cache[rel_path]
        else:
            # Fallback load
            full_path = os.path.join(self.path_config.input_dir, rel_path)
            raw_wav, _ = torchaudio.load(full_path)

        # Processing
        if is_background:
            # If this sample represents the 'silence' class and comes from a long file,
            # we must take a RANDOM crop for training to generate variety.
            # For val/test, we should ideally be consistent, but here we likely want
            # to evaluate on a valid silence clip.
            target_len = self.audio_config.sample_rate * self.audio_config.duration
            if raw_wav.shape[1] > target_len:
                if self.split == "train":
                    start = random.randint(0, raw_wav.shape[1] - target_len)
                else:
                    # Deterministic crop for validation (center)
                    start = (raw_wav.shape[1] - target_len) // 2
                wav = raw_wav[:, start : start + target_len]
            else:
                wav = self._pad_or_crop(raw_wav)
        else:
            # Standard command file
            wav = self._pad_or_crop(raw_wav)

        return wav, row["label"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform, label_str = self._get_waveform(idx)

        # 1. Waveform Augmentation (Noise Mixing)
        # Only apply if training, and NOT if the label is already silence/background
        if self.transform and label_str != "silence":
            if random.random() < self.train_config.mix_noise_prob:
                waveform = self._mix_background_noise(waveform)

        # 2. Compute Spectrogram (High-Fidelity)
        # Input: (1, 16000) -> Output: (1, n_mels, time)
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 3. Instance Normalization
        # (spec - mean) / std
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 4. Spectrogram Augmentation (SpecAugment)
        if self.transform:
            if random.random() < self.train_config.spec_augment_prob:
                spec = self.time_masking(spec)
                spec = self.freq_masking(spec)

        # 5. Label Encoding
        label_idx = self.label_to_idx.get(label_str, self.label_to_idx["unknown"])

        return spec, label_idx

    def get_sampler_weights(self) -> List[float]:
        """
        Computes weights for WeightedRandomSampler to handle class imbalance.
        """
        # Get all labels
        labels = self.df["label"].tolist()

        # Count classes
        class_counts = pd.Series(labels).value_counts()

        # Calculate weight per class: 1.0 / count
        class_weights = 1.0 / class_counts

        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in labels]

        return sample_weights


def get_dataloaders(batch_size: int = 32, num_workers: int = 4, debug: bool = False):
    """
    Factory function to create Train and Val DataLoaders.
    """
    # Train Set
    train_dataset = SpeechCommandsDataset(
        split="train", transform=True, cache_data=True, debug=debug
    )

    # Sampler for class imbalance
    weights = train_dataset.get_sampler_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set
    val_dataset = SpeechCommandsDataset(
        split="val", transform=False, cache_data=True, debug=debug
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
