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


class BackgroundNoiseAugment(nn.Module):
    """
    Applies real background noise injection.
    Cite solution_lesson_node_00055: Data realism beats algorithmic complexity.
    """

    def __init__(
        self,
        noises: torch.Tensor,
        min_snr_db: float = 10.0,
        max_snr_db: float = 30.0,
        p: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("noises", noises)
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
        if not self.training:
            return waveform

        if torch.rand(1).item() > self.p:
            return waveform

        batch_size, time_steps = waveform.shape
        device = waveform.device
        noise_len = self.noises.shape[0]

        # Select random noise segments
        max_start = noise_len - time_steps
        if max_start <= 0:
            return waveform

        start_indices = torch.randint(0, max_start, (batch_size,), device=device)

        # Vectorized gather: (Batch, 1) + (1, Time) -> (Batch, Time)
        indices = start_indices.unsqueeze(1) + torch.arange(
            time_steps, device=device
        ).unsqueeze(0)
        noise_clips = self.noises[indices]

        # Calculate powers
        sig_power = waveform.pow(2).mean(dim=1, keepdim=True)
        noise_power = noise_clips.pow(2).mean(dim=1, keepdim=True)

        # Avoid div by zero
        sig_power = torch.clamp(sig_power, min=1e-9)
        noise_power = torch.clamp(noise_power, min=1e-9)

        # Random SNR
        snr_db = torch.empty(batch_size, 1, device=device).uniform_(
            self.min_snr_db, self.max_snr_db
        )

        # Target noise power
        target_noise_power = sig_power / (10 ** (snr_db / 10))

        # Scale factor
        scale = torch.sqrt(target_noise_power / noise_power)

        return waveform + noise_clips * scale


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
