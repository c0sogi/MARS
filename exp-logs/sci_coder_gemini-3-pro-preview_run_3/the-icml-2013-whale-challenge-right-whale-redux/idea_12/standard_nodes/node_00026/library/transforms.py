import torch
import torch.nn as nn
import torchaudio
import numpy as np
from library.config import Config


class LogMelSpectrogram(nn.Module):
    """
    Transform to convert waveform to Log-Mel Spectrogram.
    Generates high-fidelity spectrograms based on configuration parameters.
    """

    def __init__(
        self,
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    ):
        super(LogMelSpectrogram, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Audio waveform of shape (..., time).
        Returns:
            torch.Tensor: Log-Mel Spectrogram of shape (..., n_mels, time).
        """
        # Ensure input is a tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Move to same device as module parameters if necessary
        try:
            device = next(self.parameters()).device
            if waveform.device != device:
                waveform = waveform.to(device)
        except StopIteration:
            pass  # Module has no parameters

        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)
        return spec


class InstanceNorm(nn.Module):
    """
    Applies Instance-level Min-Max Normalization.
    Normalizes the input tensor to the range [0, 1] based on its own min and max values.
    This preserves the spectral contrast within the sample.
    """

    def __init__(self, eps=1e-6):
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (e.g., spectrogram).
        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Compute min and max of the entire sample
        min_val = x.min()
        max_val = x.max()

        if (max_val - min_val) > self.eps:
            x = (x - min_val) / (max_val - min_val)
        else:
            x = torch.zeros_like(x)

        return x


class SpecAugment(nn.Module):
    """
    Applies SpecAugment (Time and Frequency Masking).
    Used for data augmentation during training to improve robustness.
    """

    def __init__(
        self,
        time_mask_param=Config.SPECAUG_TIME_MASK,
        freq_mask_param=Config.SPECAUG_FREQ_MASK,
        num_masks=1,
    ):
        super(SpecAugment, self).__init__()
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        self.num_masks = num_masks

    def forward(self, spec):
        """
        Args:
            spec (torch.Tensor): Spectrogram of shape (channels, freq, time).
        Returns:
            torch.Tensor: Masked spectrogram.
        """
        # Apply masks sequentially
        # torchaudio masking transforms handle multi-channel inputs (C, F, T)
        for _ in range(self.num_masks):
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        return spec
