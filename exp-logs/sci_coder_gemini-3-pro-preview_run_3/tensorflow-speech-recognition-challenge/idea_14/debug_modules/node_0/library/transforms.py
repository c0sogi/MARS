import torch
import torch.nn as nn
import torchaudio
import os
import random
import glob
import math
from typing import List

from library.config import PathConfig, AudioConfig, MelConfig, TrainConfig


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
        """Loads all background noise files into a list of tensors."""
        if not os.path.exists(self.noise_dir):
            print(f"Warning: Noise directory {self.noise_dir} not found.")
            return

        noise_files = glob.glob(os.path.join(self.noise_dir, "*.wav"))
        for f in noise_files:
            try:
                # Load noise file
                waveform, sr = torchaudio.load(f)
                # Resample if necessary
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

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
            # center=True ensures the number of time frames is consistent across resolutions
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config.sample_rate,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=mel_config.hop_length,
                n_mels=mel_config.n_mels,
                f_min=mel_config.f_min,
                f_max=mel_config.f_max,
                center=True,
                power=2.0,
            )
            self.transforms.append(transform)

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

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
