import torch
import torch.nn as nn
import torchaudio
import math
from library.config import AUDIO_CONFIG


class LogMelSpectrogram(nn.Module):
    """
    Converts raw waveforms to Log-Mel Spectrograms using GPU-accelerated torchaudio transforms.
    Output shape: (Batch, 1, n_mels, time_steps)
    """

    def __init__(self, config=AUDIO_CONFIG):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )
        # Convert power to dB (log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (Batch, Time)
        Returns:
            spectrogram: Tensor of shape (Batch, 1, Freq, Time)
        """
        # Compute Mel Spectrogram
        # Input: (Batch, Time) -> Output: (Batch, n_mels, Time)
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to Log Scale (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Add channel dimension for CNN compatibility
        # (Batch, n_mels, Time) -> (Batch, 1, n_mels, Time)
        return log_mel_spec.unsqueeze(1)


class WaveformAugment(nn.Module):
    """
    Applies additive Gaussian noise to the waveform based on a random SNR.
    """

    def __init__(
        self, min_snr_db: float = 10.0, max_snr_db: float = 30.0, p: float = 0.5
    ):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (Batch, Time)
        Returns:
            Noisy waveform: Tensor of shape (Batch, Time)
        """
        # Only apply augmentation during training (assuming caller handles mode,
        # but we check self.training for safety if model.eval() is called)
        if not self.training:
            return waveform

        # Apply with probability p
        if torch.rand(1).item() > self.p:
            return waveform

        batch_size, time_steps = waveform.shape
        device = waveform.device

        # Calculate signal power: mean(x^2) along time axis
        # Shape: (Batch, 1)
        sig_power = waveform.pow(2).mean(dim=1, keepdim=True)

        # Avoid division by zero for silent signals
        sig_power = torch.clamp(sig_power, min=1e-9)

        # Generate random SNR values for each sample in the batch
        snr_db = torch.empty(batch_size, 1, device=device).uniform_(
            self.min_snr_db, self.max_snr_db
        )

        # Calculate noise power required
        # SNR_db = 10 * log10(P_signal / P_noise)
        # => P_noise = P_signal / 10^(SNR_db / 10)
        noise_power = sig_power / (10 ** (snr_db / 10))

        # Generate noise
        # Standard normal distribution N(0, 1)
        noise = torch.randn_like(waveform)

        # Scale noise to match required power
        # Expected power of N(0,1) is 1. We multiply by sqrt(target_power).
        scaled_noise = noise * torch.sqrt(noise_power)

        return waveform + scaled_noise


class SpecAugment(nn.Module):
    """
    Applies Time and Frequency Masking to the Spectrogram.
    """

    def __init__(
        self, freq_mask_param: int = 15, time_mask_param: int = 35, p: float = 0.5
    ):
        super().__init__()
        self.p = p
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor of shape (Batch, 1, Freq, Time)
        Returns:
            Masked spectrogram
        """
        if not self.training:
            return spec

        # Apply with probability p
        if torch.rand(1).item() > self.p:
            return spec

        # Torchaudio transforms handle (..., Freq, Time)
        # We can pass the whole batch.

        # Apply Frequency Masking
        # Masking transforms usually work in-place or return new tensor.
        # Torchaudio implementation is not in-place by default.
        masked_spec = self.freq_mask(spec)

        # Apply Time Masking
        masked_spec = self.time_mask(masked_spec)

        return masked_spec
