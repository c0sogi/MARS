import os
import random
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from library.config import (
    BACKGROUND_NOISE_DIR,
    SAMPLE_RATE,
    N_MELS,
    HOP_LENGTHS,
    WIN_LENGTHS,
    F_MIN,
    F_MAX,
)


class AudioProcessor(nn.Module):
    """
    GPU-accelerated Audio Processor for Hybrid 1D-2D Stream.
    Handles dynamic background noise injection, multi-resolution spectrogram generation,
    and SpecAugment.
    """

    def __init__(self, noise_prob=0.5, snr_min=5, snr_max=30):
        super().__init__()
        self.noise_prob = noise_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.sample_rate = SAMPLE_RATE

        # 1. Load Background Noise
        # We load these into CPU memory first, they will be moved to GPU on-the-fly
        # or we can store them as buffers if we want them to persist on device.
        # Given the small size (~10MB), keeping them in a list is efficient.
        self.noises = []
        if os.path.exists(BACKGROUND_NOISE_DIR):
            files = glob.glob(os.path.join(BACKGROUND_NOISE_DIR, "*.wav"))
            for f in files:
                try:
                    # Load and ensure correct sample rate
                    waveform, sr = torchaudio.load(f)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    self.noises.append(waveform)
                except Exception as e:
                    # Ignore unreadable files
                    pass

        # 2. Define Multi-Resolution Spectrogram Transforms
        # We create a ModuleList of transforms corresponding to the config
        self.mel_transforms = nn.ModuleList()
        for win, hop in zip(WIN_LENGTHS, HOP_LENGTHS):
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=win,  # Use window length as FFT size
                win_length=win,
                hop_length=hop,
                n_mels=N_MELS,
                f_min=F_MIN,
                f_max=F_MAX,
                normalized=True,
            )
            self.mel_transforms.append(mel)

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        # 3. Define SpecAugment Transforms
        # Parameters tuned for 1-second clips (approx 50-100 time steps depending on hop)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)

    def mix_background_noise(self, waveforms):
        """
        Injects background noise into the waveforms.
        Args:
            waveforms (torch.Tensor): Input batch of shape (B, T).
        Returns:
            torch.Tensor: Augmented batch of shape (B, T).
        """
        if not self.noises:
            return waveforms

        B = waveforms.shape[0]
        device = waveforms.device

        # Work with (B, 1, T) for easier broadcasting logic
        if waveforms.dim() == 2:
            augmented = waveforms.unsqueeze(1).clone()
        else:
            augmented = waveforms.clone()

        for i in range(B):
            # Apply noise with probability noise_prob
            if random.random() < self.noise_prob:
                # Select random noise file
                noise = random.choice(self.noises).to(device)

                # Noise shape is (C, T_noise), usually C=1
                noise_len = noise.shape[-1]
                sig_len = augmented.shape[-1]

                # Crop noise to match signal length
                if noise_len > sig_len:
                    start = random.randint(0, noise_len - sig_len)
                    noise_crop = noise[:, start : start + sig_len]
                else:
                    # Pad noise if shorter
                    repeats = math.ceil(sig_len / noise_len)
                    noise_crop = noise.repeat(1, repeats)[:, :sig_len]

                # Calculate energies
                sig_energy = augmented[i].pow(2).mean()
                noise_energy = noise_crop.pow(2).mean()

                # Avoid division by zero
                if noise_energy > 1e-9:
                    # Determine random SNR
                    target_snr = random.uniform(self.snr_min, self.snr_max)
                    snr_factor = 10 ** (target_snr / 10)

                    # Calculate scaling factor
                    target_noise_energy = sig_energy / snr_factor
                    scale = torch.sqrt(target_noise_energy / noise_energy)

                    # Add noise
                    augmented[i] = augmented[i] + scale * noise_crop

        # Squeeze back to (B, T) if input was (B, T)
        if waveforms.dim() == 2:
            return augmented.squeeze(1)
        return augmented

    def compute_multires_spectrogram(self, waveforms):
        """
        Computes 3-channel multi-resolution spectrograms.
        Resizes all spectrograms to match the temporal dimension of the highest resolution one.
        Args:
            waveforms (torch.Tensor): Input batch (B, T).
        Returns:
            torch.Tensor: Stacked spectrograms (B, 3, F, T).
        """
        specs = []
        target_width = None

        for i, transform in enumerate(self.mel_transforms):
            # Compute Mel Spectrogram: (B, F, T)
            spec = transform(waveforms)
            spec = self.amp_to_db(spec)

            if i == 0:
                # The first transform uses the shortest hop (320), resulting in the longest time dimension.
                # We use this as the target width for alignment.
                target_width = spec.shape[-1]
                specs.append(spec)
            else:
                # Resize lower-resolution spectrograms to match the target width.
                # F.interpolate expects (N, C, H, W) or (N, C, L).
                # We treat F as H and T as W. Unsqueeze to (B, 1, F, T).
                spec_in = spec.unsqueeze(1)
                spec_resized = F.interpolate(
                    spec_in,
                    size=(spec.shape[-2], target_width),  # Keep F same, resize T
                    mode="bilinear",
                    align_corners=False,
                )
                specs.append(spec_resized.squeeze(1))

        # Stack along channel dimension: (B, 3, F, T)
        return torch.stack(specs, dim=1)

    def apply_spec_augment(self, specs):
        """
        Applies SpecAugment (Time and Frequency Masking).
        Args:
            specs (torch.Tensor): Input spectrograms (B, C, F, T).
        Returns:
            torch.Tensor: Masked spectrograms.
        """
        B, C, F_dim, T_dim = specs.shape

        # Reshape to (B*C, F, T) to apply masking independently to each channel.
        # This prevents the model from relying on a single channel if one is masked.
        specs_flat = specs.view(B * C, F_dim, T_dim)

        # Apply masks
        specs_aug = self.freq_mask(specs_flat)
        specs_aug = self.time_mask(specs_aug)

        # Reshape back
        return specs_aug.view(B, C, F_dim, T_dim)

    def forward(self, waveforms):
        """
        Forward pass of the processor.
        Args:
            waveforms (torch.Tensor): Raw audio batch (B, T).
        Returns:
            tuple: (waveforms_out, specs_out)
                - waveforms_out: (B, T) Augmented raw audio (for 1D stream).
                - specs_out: (B, 3, F, T) Augmented spectrograms (for 2D stream).
        """
        # 1. Background Noise Injection (Only during training)
        if self.training:
            waveforms_out = self.mix_background_noise(waveforms)
        else:
            waveforms_out = waveforms

        # 2. Feature Extraction (Spectrograms)
        # Note: We compute specs from the potentially noisy waveform
        specs_out = self.compute_multires_spectrogram(waveforms_out)

        # 3. SpecAugment (Only during training)
        if self.training:
            specs_out = self.apply_spec_augment(specs_out)

        return waveforms_out, specs_out
