import os
import random
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from library.config import (
    BACKGROUND_NOISE_DIR,
    SAMPLE_RATE,
    N_MELS,
    HOP_LENGTHS,
    WIN_LENGTHS,
    F_MIN,
    F_MAX,
)


def get_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """
    Creates a Mel filterbank matrix using NumPy.
    Returns: (n_mels, n_fft // 2 + 1)
    """
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)

    mel_min = 2595 * np.log10(1 + fmin / 700.0)
    mel_max = 2595 * np.log10(1 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)

    hz_points = 700 * (10 ** (mels / 2595.0) - 1)

    weights = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]

        lower_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if lower_mask.any():
            weights[i, lower_mask] = (fft_freqs[lower_mask] - f_left) / (
                f_center - f_left
            )

        upper_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if upper_mask.any():
            weights[i, upper_mask] = (f_right - fft_freqs[upper_mask]) / (
                f_right - f_center
            )

    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    weights *= enorm[:, np.newaxis]

    return torch.from_numpy(weights).float()


class AudioProcessor(nn.Module):
    """
    GPU-accelerated Audio Processor for Hybrid 1D-2D Stream.
    Handles dynamic background noise injection, multi-resolution spectrogram generation,
    and SpecAugment using native Torch/NumPy operations to avoid binary incompatibility.
    """

    def __init__(self, noise_prob=0.5, snr_min=5, snr_max=30):
        super().__init__()
        self.noise_prob = noise_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.sample_rate = SAMPLE_RATE

        # 1. Load Background Noise
        self.noises = []
        if os.path.exists(BACKGROUND_NOISE_DIR):
            files = glob.glob(os.path.join(BACKGROUND_NOISE_DIR, "*.wav"))
            for f in files:
                try:
                    # Load using soundfile
                    waveform, sr = sf.read(f)
                    waveform = torch.from_numpy(waveform).float()

                    # Ensure (Channels, Time) format
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)  # (1, T)
                    elif waveform.dim() == 2:
                        waveform = waveform.t()  # (C, T)

                    # Resample if needed
                    if sr != self.sample_rate:
                        new_len = int(waveform.shape[-1] * (self.sample_rate / sr))
                        waveform = F.interpolate(
                            waveform.unsqueeze(0),
                            size=new_len,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0)

                    self.noises.append(waveform)
                except Exception:
                    pass

        # 2. Define Multi-Resolution Spectrogram Parameters
        # Pre-compute Mel matrices and register as buffers
        for i, (win, hop) in enumerate(zip(WIN_LENGTHS, HOP_LENGTHS)):
            mel_basis = get_mel_filterbank(self.sample_rate, win, N_MELS, F_MIN, F_MAX)
            self.register_buffer(f"mel_basis_{i}", mel_basis)

            window = torch.hann_window(win)
            self.register_buffer(f"window_{i}", window)

        # 3. SpecAugment Parameters
        self.freq_mask_param = 15
        self.time_mask_param = 10

    def mix_background_noise(self, waveforms):
        """
        Injects background noise into the waveforms.
        Args:
            waveforms (torch.Tensor): Input batch of shape (B, T).
        """
        if not self.noises:
            return waveforms

        B = waveforms.shape[0]
        device = waveforms.device

        if waveforms.dim() == 2:
            augmented = waveforms.unsqueeze(1).clone()
        else:
            augmented = waveforms.clone()

        for i in range(B):
            if random.random() < self.noise_prob:
                noise = random.choice(self.noises).to(device)

                noise_len = noise.shape[-1]
                sig_len = augmented.shape[-1]

                if noise_len > sig_len:
                    start = random.randint(0, noise_len - sig_len)
                    noise_crop = noise[:, start : start + sig_len]
                else:
                    repeats = math.ceil(sig_len / noise_len)
                    noise_crop = noise.repeat(1, repeats)[:, :sig_len]

                sig_energy = augmented[i].pow(2).mean()
                noise_energy = noise_crop.pow(2).mean()

                if noise_energy > 1e-9:
                    target_snr = random.uniform(self.snr_min, self.snr_max)
                    snr_factor = 10 ** (target_snr / 10)
                    target_noise_energy = sig_energy / snr_factor
                    scale = torch.sqrt(target_noise_energy / noise_energy)
                    augmented[i] = augmented[i] + scale * noise_crop

        if waveforms.dim() == 2:
            return augmented.squeeze(1)
        return augmented

    def compute_spectrogram(self, waveform, idx):
        """
        Computes Mel Spectrogram using native torch.stft and pre-computed mel basis.
        """
        win_length = WIN_LENGTHS[idx]
        hop_length = HOP_LENGTHS[idx]
        mel_basis = getattr(self, f"mel_basis_{idx}")
        window = getattr(self, f"window_{idx}")

        # STFT
        stft = torch.stft(
            waveform,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        mag = torch.abs(stft)
        mel_spec = torch.matmul(mel_basis, mag)

        # Amplitude to DB
        mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))

        return mel_spec

    def compute_multires_spectrogram(self, waveforms):
        """
        Computes 3-channel multi-resolution spectrograms.
        """
        specs = []
        target_width = None

        for i in range(len(WIN_LENGTHS)):
            spec = self.compute_spectrogram(waveforms, i)

            if i == 0:
                target_width = spec.shape[-1]
                specs.append(spec)
            else:
                spec_in = spec.unsqueeze(1)
                spec_resized = F.interpolate(
                    spec_in,
                    size=(spec.shape[-2], target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                specs.append(spec_resized.squeeze(1))

        return torch.stack(specs, dim=1)

    def apply_spec_augment(self, specs):
        """
        Applies SpecAugment (Time and Frequency Masking) manually.
        """
        if not self.training:
            return specs

        B, C, F_dim, T_dim = specs.shape
        augmented = specs.clone()

        for b in range(B):
            for c in range(C):
                # Frequency Masking
                f_mask_len = random.randint(0, self.freq_mask_param)
                if f_mask_len > 0:
                    f_start = random.randint(0, F_dim - f_mask_len)
                    augmented[b, c, f_start : f_start + f_mask_len, :] = 0.0

                # Time Masking
                t_mask_len = random.randint(0, self.time_mask_param)
                if t_mask_len > 0:
                    t_start = random.randint(0, T_dim - t_mask_len)
                    augmented[b, c, :, t_start : t_start + t_mask_len] = 0.0

        return augmented

    def forward(self, waveforms):
        if self.training:
            waveforms_out = self.mix_background_noise(waveforms)
        else:
            waveforms_out = waveforms

        specs_out = self.compute_multires_spectrogram(waveforms_out)

        if self.training:
            specs_out = self.apply_spec_augment(specs_out)

        return waveforms_out, specs_out
