import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import os
import random
import glob
import math
from typing import List

from library.config import PathConfig, AudioConfig, MelConfig, TrainConfig


# --- Helper Functions for Mel Filterbank (Replacing torchaudio) ---


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def get_mel_basis(sr, n_fft, n_mels, fmin, fmax):
    """
    Creates a Mel filterbank matrix using NumPy.
    Returns: (n_mels, n_freqs)
    """
    if fmax is None:
        fmax = sr / 2.0

    n_freqs = n_fft // 2 + 1

    # Create Mel scale points
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)

    # Map to FFT bins
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

    weights = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        b_prev = bins[i]
        b_curr = bins[i + 1]
        b_next = bins[i + 2]

        # Up slope
        for j in range(b_prev, b_curr):
            weights[i, j] = (j - b_prev) / (b_curr - b_prev)

        # Down slope
        for j in range(b_curr, b_next):
            # Avoid index out of bounds if b_next exceeds n_freqs
            if j < n_freqs:
                weights[i, j] = (b_next - j) / (b_next - b_curr)

    return weights


class CustomMelSpectrogram(nn.Module):
    """
    Native PyTorch implementation of MelSpectrogram to avoid torchaudio dependency.
    """

    def __init__(
        self, sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Create Mel Basis
        mel_basis_np = get_mel_basis(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis_np).float())

        # Window function (Hann)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) waveform
        Returns:
            mel_spec: (B, 1, n_mels, T_frames)
        """
        # Remove channel dim for STFT: (B, T)
        x_squeezed = x.squeeze(1)

        # STFT
        # center=True pads the input so that the t-th frame is centered at time t * hop_length
        stft = torch.stft(
            x_squeezed,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        # Power Spectrogram: |STFT|^2
        power_spec = stft.abs().pow(2.0)  # (B, n_freqs, T_frames)

        # Apply Mel Basis
        # (n_mels, n_freqs) @ (B, n_freqs, T_frames) -> (B, n_mels, T_frames)
        mel_spec = torch.matmul(self.mel_basis, power_spec)

        # Add channel dim back
        return mel_spec.unsqueeze(1)


class AmplitudeToDB(nn.Module):
    """
    Native PyTorch implementation of AmplitudeToDB.
    """

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        # x is power spectrogram
        # 10 * log10(x)
        x_db = 10.0 * torch.log10(torch.clamp(x, min=1e-10))

        # Peak normalization
        max_val = x_db.amax(dim=(1, 2, 3), keepdim=True)
        return torch.clamp(x_db, min=max_val - self.top_db)


class GPUNoiseInjector(nn.Module):
    """
    Dynamically mixes background noise into speech waveforms on the GPU.
    Loads noise files into memory and mixes them on-the-fly during the forward pass.
    """

    def __init__(
        self,
        path_config: PathConfig,
        audio_config: AudioConfig,
        train_config: TrainConfig,
    ):
        super().__init__()
        self.noise_dir = path_config.noise_dir
        self.sample_rate = audio_config.sample_rate
        self.noise_prob = train_config.noise_prob
        self.snr_min = train_config.noise_snr_min
        self.snr_max = train_config.noise_snr_max

        self.noises = []
        self._load_noises()

    def _load_noises(self):
        """Loads all background noise files into a list of tensors using soundfile."""
        if not os.path.exists(self.noise_dir):
            print(f"Warning: Noise directory {self.noise_dir} not found.")
            return

        noise_files = glob.glob(os.path.join(self.noise_dir, "*.wav"))
        for f in noise_files:
            try:
                # Load noise file using soundfile
                # sf.read returns (samples, channels) or (samples,)
                data, sr = sf.read(f, dtype="float32")

                # Convert to Tensor (Channels, Time)
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)
                else:
                    waveform = torch.from_numpy(data).t()  # (C, T)

                # Resample if necessary using native torch interpolation
                if sr != self.sample_rate:
                    # Interpolate expects (B, C, T)
                    waveform = waveform.unsqueeze(0)
                    new_len = int(waveform.shape[-1] * (self.sample_rate / sr))
                    waveform = F.interpolate(
                        waveform, size=new_len, mode="linear", align_corners=False
                    )
                    waveform = waveform.squeeze(0)

                # Store on CPU to conserve GPU memory; move to GPU only when used
                self.noises.append(waveform)
            except Exception as e:
                print(f"Error loading noise file {f}: {e}")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, 1, T) raw audio tensor.
        Returns:
            augmented_waveforms: (B, 1, T) tensor with noise injected.
        """
        # Apply only during training and if noises exist
        if not self.training or not self.noises or self.noise_prob <= 0.0:
            return waveforms

        B, C, T = waveforms.shape
        device = waveforms.device

        # Clone to avoid in-place modification of the original batch
        augmented = waveforms.clone()

        for i in range(B):
            if random.random() < self.noise_prob:
                # Select a random noise clip
                noise_idx = random.randint(0, len(self.noises) - 1)
                noise = self.noises[noise_idx]  # (C_noise, T_noise)

                # Ensure noise is mono
                if noise.shape[0] > 1:
                    noise = torch.mean(noise, dim=0, keepdim=True)

                # Move selected noise to the same device as input
                noise = noise.to(device)
                noise_len = noise.shape[-1]

                # Handle noise length
                if noise_len < T:
                    # Tile noise if it's shorter than the input
                    repeats = math.ceil(T / noise_len)
                    noise = noise.repeat(1, repeats)
                    noise_len = noise.shape[-1]

                # Random crop
                start = random.randint(0, noise_len - T)
                noise_crop = noise[:, start : start + T]  # (1, T)

                # Calculate Signal and Noise Power
                sig_power = augmented[i].pow(2).mean()
                noise_power = noise_crop.pow(2).mean()

                # Mix if noise is not silent
                if noise_power > 1e-9:
                    target_snr_db = random.uniform(self.snr_min, self.snr_max)
                    target_ratio = 10 ** (target_snr_db / 10.0)

                    # Calculate scaling factor
                    # SNR = P_signal / P_noise_added
                    # P_noise_added = (scale * noise)^2 = scale^2 * P_noise
                    # scale = sqrt(P_signal / (SNR * P_noise))

                    if sig_power > 1e-9:
                        scale = torch.sqrt(sig_power / (target_ratio * noise_power))
                        augmented[i] = augmented[i] + scale * noise_crop

                # Clamp to valid audio range [-1, 1]
                augmented[i] = torch.clamp(augmented[i], -1.0, 1.0)

        return augmented


class MultiResMelSpectrogram(nn.Module):
    """
    Computes 3-Channel Multi-Resolution Log-Mel Spectrograms.
    Stacks spectrograms computed with Short, Medium, and Long windows.
    """

    def __init__(self, mel_config: MelConfig, audio_config: AudioConfig):
        super().__init__()

        self.transforms = nn.ModuleList()

        # Create a MelSpectrogram transform for each resolution defined in config
        for win_len, n_fft in zip(mel_config.win_lengths, mel_config.n_ffts):
            transform = CustomMelSpectrogram(
                sample_rate=audio_config.sample_rate,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=mel_config.hop_length,
                n_mels=mel_config.n_mels,
                f_min=mel_config.f_min,
                f_max=mel_config.f_max,
            )
            self.transforms.append(transform)

        self.amp_to_db = AmplitudeToDB(top_db=80)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, 1, T) raw audio.
        Returns:
            multi_res_spec: (B, 3, F, T) log-mel spectrograms.
        """
        specs = []
        for t in self.transforms:
            # Compute spec: (B, 1, F, T)
            spec = t(waveforms)
            specs.append(spec)

        # Concatenate along the channel dimension
        # Result: (B, 3, F, T)
        multi_res_spec = torch.cat(specs, dim=1)

        # Convert to Log Scale (dB)
        multi_res_spec = self.amp_to_db(multi_res_spec)

        return multi_res_spec


class GPUSpecAugment(nn.Module):
    """
    Applies Frequency and Time Masking on the GPU.
    Fills masked regions with the minimum value of the instance.
    """

    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.prob = train_config.spec_aug_prob
        self.freq_mask_param = train_config.freq_mask_param
        self.time_mask_param = train_config.time_mask_param

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            specs: (B, C, F, T) spectrograms.
        Returns:
            augmented_specs: (B, C, F, T) masked spectrograms.
        """
        if not self.training or self.prob <= 0.0:
            return specs

        B, C, F, T = specs.shape
        augmented = specs.clone()

        for i in range(B):
            if random.random() < self.prob:
                # Find minimum value for this instance to use as fill value
                min_val = augmented[i].min()

                # Frequency Masking
                # Apply mask to all channels simultaneously to preserve structure
                f_width = random.randint(0, self.freq_mask_param)
                f_start = random.randint(0, max(0, F - f_width))

                # Mask: [:, f_start:f_start+f_width, :]
                augmented[i, :, f_start : f_start + f_width, :] = min_val

                # Time Masking
                t_width = random.randint(0, self.time_mask_param)
                t_start = random.randint(0, max(0, T - t_width))

                # Mask: [:, :, t_start:t_start+t_width]
                augmented[i, :, :, t_start : t_start + t_width] = min_val

        return augmented
