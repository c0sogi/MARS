import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
import cv2
import random
from typing import Dict, Optional, Tuple, Union, List

from library.config import Config

# ==========================================
# Augmentation Utilities
# ==========================================


class SpecAugment:
    """
    Applies Time and Frequency Masking to a spectrogram.
    Supports input shape (Channels, Freq, Time).
    """

    def __init__(
        self,
        num_mask=2,
        freq_masking_max_percentage=0.15,
        time_masking_max_percentage=0.20,
        prob=0.5,
    ):
        self.num_mask = num_mask
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.time_masking_max_percentage = time_masking_max_percentage
        self.prob = prob

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Args:
            spec: Numpy array of shape (C, F, T) or (F, T)
        Returns:
            Masked spectrogram
        """
        if random.random() > self.prob:
            return spec

        spec = spec.copy()

        # Ensure shape is (C, F, T)
        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]

        C, F, T = spec.shape

        # Frequency Masking
        for _ in range(self.num_mask):
            f_max = int(F * self.freq_masking_max_percentage)
            if f_max > 0:
                f = random.randint(0, f_max)
                f0 = random.randint(0, F - f)
                spec[:, f0 : f0 + f, :] = 0.0

        # Time Masking
        for _ in range(self.num_mask):
            t_max = int(T * self.time_masking_max_percentage)
            if t_max > 0:
                t = random.randint(0, t_max)
                t0 = random.randint(0, T - t)
                spec[:, :, t0 : t0 + t] = 0.0

        return spec


class MixUp:
    """
    Applies MixUp augmentation to a batch of data.
    """

    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self, batch_data: Dict[str, torch.Tensor], batch_targets: torch.Tensor
    ):
        """
        Args:
            batch_data: Dict containing 'eeg' and 'spec' tensors.
            batch_targets: Tensor of shape (Batch, NumClasses)
        Returns:
            mixed_data, mixed_targets
        """
        if random.random() > self.prob:
            return batch_data, batch_targets

        batch_size = batch_targets.size(0)
        indices = torch.randperm(batch_size).to(batch_targets.device)

        lam = np.random.beta(self.alpha, self.alpha)

        # Mix Inputs
        mixed_data = {}
        for key, data in batch_data.items():
            mixed_data[key] = lam * data + (1 - lam) * data[indices]

        # Mix Targets
        mixed_targets = lam * batch_targets + (1 - lam) * batch_targets[indices]

        return mixed_data, mixed_targets


# ==========================================
# Transformation Classes
# ==========================================


class EEGTransform:
    """
    Processes raw EEG DataFrame into Siamese Spectrogram Inputs.
    Output Shape: (4, 5, H, W) -> 4 Views, 5 Channels each.
    """

    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.img_size = Config.IMG_SIZE

        # MelSpectrogram Configuration
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.EEG_SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=0,
            f_max=None,
            center=True,
        )

        # Normalization Stats (ImageNet) extended to 5 channels
        # Base: [0.485, 0.456, 0.406]
        self.mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225, 0.229, 0.224], dtype=np.float32)

        self.spec_augment = SpecAugment(prob=0.5) if mode == "train" else None

    def _compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Computes Log-Mel Spectrogram for a 1D signal."""
        # Signal shape: (T,)
        tensor_sig = torch.from_numpy(signal).float()
        # MelSpec shape: (n_mels, time)
        spec = self.mel_transform(tensor_sig)
        # Log transform (add epsilon for stability)
        spec = torch.log(spec + 1e-6)
        return spec.numpy()

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resizes (C, H, W) to (C, TargetH, TargetW)."""
        # Transpose to (H, W, C) for cv2
        img_hwc = np.transpose(img, (1, 2, 0))
        img_resized = cv2.resize(
            img_hwc,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Handle case where C=1 and cv2 removes the dimension
        if img_resized.ndim == 2:
            img_resized = img_resized[..., np.newaxis]

        # Transpose back to (C, H, W)
        return np.transpose(img_resized, (2, 0, 1))

    def __call__(self, eeg_df: pd.DataFrame) -> torch.Tensor:
        """
        Args:
            eeg_df: DataFrame with EEG data (19 columns).
        Returns:
            Tensor of shape (4, 5, H, W)
        """
        # 1. Fill NaNs
        eeg_df = eeg_df.fillna(0)

        views = []
        chain_names = ["LL", "RL", "LP", "RP"]

        for chain_key in chain_names:
            electrodes = Config.CHAIN_CONFIG[chain_key]
            chain_specs = []

            for elec in electrodes:
                if elec in eeg_df.columns:
                    sig = eeg_df[elec].values
                else:
                    sig = np.zeros(len(eeg_df))

                # Compute Spec: (Mel, Time)
                s = self._compute_spectrogram(sig)
                chain_specs.append(s)

            # Stack electrodes depth-wise: (5, Mel, Time)
            view_img = np.stack(chain_specs, axis=0)

            # Resize
            view_img = self._resize_image(view_img)

            # Augment (SpecAugment)
            if self.spec_augment:
                view_img = self.spec_augment(view_img)

            # Normalize
            # view_img is (5, H, W)
            view_img = (view_img - self.mean[:, None, None]) / self.std[:, None, None]

            views.append(view_img)

        # Stack views: (4, 5, H, W)
        output = np.stack(views, axis=0)
        return torch.from_numpy(output).float()


class SpectrogramTransform:
    """
    Processes Kaggle Spectrograms (Context Stream).
    Output Shape: (4, H, W) -> 4 Regions (LL, RL, LP, RP) as channels.
    """

    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.img_size = Config.IMG_SIZE

        # Normalization (ImageNet extended to 4 channels)
        # We'll repeat the first channel stat for the 4th
        self.mean = np.array([0.485, 0.456, 0.406, 0.485], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225, 0.229], dtype=np.float32)

        self.spec_augment = SpecAugment(prob=0.5) if mode == "train" else None

    def __call__(self, spec_arr: np.ndarray) -> torch.Tensor:
        """
        Args:
            spec_arr: Numpy array of shape (Time, Freq) or (Time, Freq*4)
                      or pre-split (4, F, T).
                      Kaggle specs are usually (Time, 401) or similar.
                      We assume the dataset passes a shape (4, Freq, Time)
                      or (Time, Freq_Total) that needs splitting.

                      Based on standard handling: Kaggle specs have 4 regions concatenated in freq.
                      We assume the Dataset handles loading and reshaping to (4, F, T).
        Returns:
            Tensor of shape (4, H, W)
        """
        # Handle NaNs
        spec_arr = np.nan_to_num(spec_arr, nan=0.0)

        # Log transform
        spec_arr = np.log(spec_arr + 1e-6)

        # Resize
        # Input is (4, F, T), need (4, H, W)
        # Transpose to (T, F, 4) or (F, T, 4) isn't right for cv2 if F!=T
        # We treat it as an image with 4 channels.
        # Current shape: (C, H_in, W_in)
        img_hwc = np.transpose(spec_arr, (1, 2, 0))
        img_resized = cv2.resize(
            img_hwc,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Back to (C, H, W)
        spec_proc = np.transpose(img_resized, (2, 0, 1))

        # Augment
        if self.spec_augment:
            spec_proc = self.spec_augment(spec_proc)

        # Normalize
        spec_proc = (spec_proc - self.mean[:, None, None]) / self.std[:, None, None]

        return torch.from_numpy(spec_proc).float()


def get_transforms(mode: str = "train") -> Dict:
    """
    Factory function to return transforms for both streams.
    """
    return {"eeg": EEGTransform(mode=mode), "spec": SpectrogramTransform(mode=mode)}
