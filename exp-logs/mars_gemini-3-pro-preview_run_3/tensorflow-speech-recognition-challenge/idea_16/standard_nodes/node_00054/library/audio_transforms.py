import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from library.config import Config


class MelSpectrogram(nn.Module):
    """
    Custom MelSpectrogram implementation using native PyTorch STFT.
    Replaces torchaudio.transforms.MelSpectrogram.
    """

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min, f_max):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Create Mel Basis
        mel_basis = self._create_mel_basis(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Create Window (Hann)
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def _create_mel_basis(self, sr, n_fft, n_mels, fmin, fmax):
        """
        Creates a triangular Mel filterbank using NumPy.
        """
        # Initialize weights
        n_freqs = int(1 + n_fft // 2)
        weights = np.zeros((n_mels, n_freqs))

        # Mel points
        mel_fmin = 2595 * np.log10(1 + fmin / 700.0)
        mel_fmax = 2595 * np.log10(1 + fmax / 700.0)

        mel_points = np.linspace(mel_fmin, mel_fmax, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)

        # Bin points
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        for i in range(n_mels):
            start = bin_points[i]
            center = bin_points[i + 1]
            end = bin_points[i + 2]

            # Left slope
            if center > start:
                weights[i, start:center] = (np.arange(start, center) - start) / (
                    center - start
                )
            # Right slope
            if end > center:
                weights[i, center:end] = (end - np.arange(center, end)) / (end - center)

        return weights

    def forward(self, x):
        # x: (Batch, Time)

        # STFT
        # Returns complex tensor: (Batch, Freq, Frames) if return_complex=True
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Power Spectrogram: |STFT|^2
        power_spec = stft.abs().pow(2.0)

        # Mel Projection: (Batch, n_mels, Frames)
        return torch.matmul(self.mel_basis, power_spec)


class AmplitudeToDB(nn.Module):
    """
    Custom AmplitudeToDB implementation.
    Replaces torchaudio.transforms.AmplitudeToDB.
    """

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        # x is power spectrogram
        # Add epsilon to avoid log(0)
        x_db = 10 * torch.log10(torch.clamp(x, min=1e-10))

        # Dynamic range compression
        max_val = x_db.amax(dim=(1, 2), keepdim=True)
        return torch.max(x_db, max_val - self.top_db)


class TimeMasking(nn.Module):
    """
    Custom TimeMasking implementation.
    Replaces torchaudio.transforms.TimeMasking.
    """

    def __init__(self, time_mask_param, p=0.5):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.p = p

    def forward(self, x):
        # x: (Batch, Channels, Freq, Time)
        if not self.training:
            return x

        # Apply mask per sample or per batch? Torchaudio does per sample usually.
        # For efficiency, we'll do a simple mask here.
        if torch.rand(1) > self.p:
            return x

        B, C, F, T = x.shape
        mask_len = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, max(0, T - mask_len))

        x_masked = x.clone()
        # Cite Lesson 10: Mask with minimum value instead of 0 for log-space spectrograms
        x_masked[..., t0 : t0 + mask_len] = x_masked.min()
        return x_masked


class FrequencyMasking(nn.Module):
    """
    Custom FrequencyMasking implementation.
    Replaces torchaudio.transforms.FrequencyMasking.
    """

    def __init__(self, freq_mask_param):
        super().__init__()
        self.freq_mask_param = freq_mask_param

    def forward(self, x):
        # x: (Batch, Channels, Freq, Time)
        if not self.training:
            return x

        B, C, F, T = x.shape
        mask_len = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, max(0, F - mask_len))

        x_masked = x.clone()
        # Cite Lesson 10: Mask with minimum value instead of 0 for log-space spectrograms
        x_masked[..., f0 : f0 + mask_len, :] = x_masked.min()
        return x_masked


class GPUAudioProcessor(nn.Module):
    """
    A GPU-native audio processing module that handles:
    1. Physics-based Augmentation (Time Stretch/Resampling, Noise Mixing) on raw waveforms.
    2. Multi-Resolution Log-Mel Spectrogram extraction (Native PyTorch).
    3. SpecAugment (Time/Freq masking).
    4. Resizing and Normalization for the vision backbone.
    """

    def __init__(self):
        super().__init__()

        self.sample_rate = Config.SAMPLE_RATE
        self.num_samples = Config.NUM_SAMPLES
        self.device = torch.device(Config.DEVICE)

        # Probabilities
        self.aug_prob = Config.AUG_PROB
        self.noise_prob = Config.NOISE_PROB

        # ---------------------------------------------------------------------
        # 1. Feature Extraction Transforms (Multi-Resolution)
        # ---------------------------------------------------------------------
        self.mel_transforms = nn.ModuleList()
        for n_fft, hop_length in zip(Config.N_FFT_LIST, Config.HOP_LENGTH_LIST):
            transform = MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=Config.N_MELS,
                f_min=Config.F_MIN,
                f_max=Config.F_MAX,
            )
            self.mel_transforms.append(transform)

        self.amp_to_db = AmplitudeToDB(top_db=80)

        # ---------------------------------------------------------------------
        # 2. SpecAugment Transforms
        # ---------------------------------------------------------------------
        self.freq_mask = FrequencyMasking(freq_mask_param=Config.MASK_FREQ_LIMIT)
        self.time_mask = TimeMasking(
            time_mask_param=Config.MASK_TIME_LIMIT, p=Config.MASK_TIME_PROB
        )

        # ---------------------------------------------------------------------
        # 3. Background Noise Loading (using soundfile)
        # ---------------------------------------------------------------------
        self.noises = []
        if os.path.exists(Config.NOISE_DIR):
            noise_files = [
                f for f in os.listdir(Config.NOISE_DIR) if f.endswith(".wav")
            ]
            for nf in noise_files:
                path = os.path.join(Config.NOISE_DIR, nf)
                try:
                    # Load using soundfile
                    wav_numpy, sr = sf.read(path)

                    # Convert to tensor
                    waveform = torch.from_numpy(wav_numpy).float()

                    # Handle stereo to mono
                    if waveform.ndim > 1:
                        waveform = waveform.mean(dim=1)  # Average channels

                    # Move to GPU
                    waveform = waveform.to(self.device)

                    # Resample if needed using interpolation
                    if sr != self.sample_rate:
                        # Reshape for interpolate: (1, 1, Time)
                        waveform = waveform.view(1, 1, -1)
                        scale_factor = self.sample_rate / sr
                        waveform = F.interpolate(
                            waveform,
                            scale_factor=scale_factor,
                            mode="linear",
                            align_corners=False,
                        )
                        waveform = waveform.view(-1)  # Flatten back

                    self.noises.append(waveform)
                except Exception as e:
                    pass  # Ignore broken files

    def _add_background_noise(self, waveforms):
        """
        Mixes background noise into the batch of waveforms.
        """
        if not self.noises:
            return waveforms

        augmented = waveforms.clone()
        batch_size = waveforms.shape[0]

        # Iterate to allow different noise selection per sample
        for i in range(batch_size):
            if torch.rand(1, device=self.device) < self.noise_prob:
                noise = random.choice(self.noises)  # (T)
                noise_len = noise.shape[0]

                # Get a random crop of the noise
                if noise_len <= self.num_samples:
                    repeats = math.ceil(self.num_samples / noise_len)
                    noise_crop = noise.repeat(repeats)[: self.num_samples]
                else:
                    start = torch.randint(
                        0, noise_len - self.num_samples, (1,), device=self.device
                    )
                    noise_crop = noise[start : start + self.num_samples]

                # Calculate scaling for random SNR
                sig_rms = augmented[i].norm(p=2)
                noise_rms = noise_crop.norm(p=2)

                if noise_rms > 1e-6:
                    # SNR between 5 and 20 dB
                    snr_db = torch.empty(1, device=self.device).uniform_(5, 20)
                    scale = (sig_rms / noise_rms) * torch.pow(10, -snr_db / 20)

                    # Add noise
                    augmented[i] = augmented[i] + (scale * noise_crop)

        return augmented

    def _apply_physics_aug(self, waveforms):
        """
        Applies Time Stretch (Resampling). Pitch Shift is removed due to lack of torchaudio.
        """
        augmented = waveforms.clone()

        # Time Stretch (Speed Perturbation via Resampling)
        if torch.rand(1, device=self.device) < self.aug_prob:
            # Speed factor: 0.9 (slow) to 1.1 (fast)
            speed = (
                torch.empty(1, device=self.device)
                .uniform_(
                    1.0 - Config.TIME_STRETCH_RATE, 1.0 + Config.TIME_STRETCH_RATE
                )
                .item()
            )

            # Resample batch using Linear Interpolation
            # Input to interpolate: (Batch, Channels, Time)
            # waveforms: (Batch, Time) -> (Batch, 1, Time)
            inp = augmented.unsqueeze(1)

            # We want to "play faster" -> fewer samples -> downsample
            # If speed=1.1, we want new_len = old_len / 1.1
            scale_factor = 1.0 / speed

            resampled = F.interpolate(
                inp, scale_factor=scale_factor, mode="linear", align_corners=False
            )
            resampled = resampled.squeeze(1)

            # Crop or Pad to maintain fixed input length
            curr_len = resampled.shape[-1]
            if curr_len > self.num_samples:
                # Center crop
                start = (curr_len - self.num_samples) // 2
                augmented = resampled[..., start : start + self.num_samples]
            elif curr_len < self.num_samples:
                # Pad end
                augmented = F.pad(resampled, (0, self.num_samples - curr_len))
            else:
                augmented = resampled

        return augmented

    def forward(self, waveforms):
        """
        Forward pass.
        Args:
            waveforms (torch.Tensor): Raw audio (Batch, Time).
        Returns:
            features (torch.Tensor): (Batch, 3, 224, 224).
        """
        # Ensure input is on the correct device
        waveforms = waveforms.to(self.device)

        # ---------------------------------------------------------------------
        # 1. Waveform Augmentation (Training Only)
        # ---------------------------------------------------------------------
        if self.training:
            waveforms = self._add_background_noise(waveforms)
            waveforms = self._apply_physics_aug(waveforms)

        # ---------------------------------------------------------------------
        # 2. Multi-Resolution Spectrogram Generation
        # ---------------------------------------------------------------------
        mels = []
        for transform in self.mel_transforms:
            # Compute Mel: (Batch, n_mels, time)
            mel = transform(waveforms)

            # Log Scale (dB)
            mel = self.amp_to_db(mel)

            # Cite Lesson 53: Avoid Interpolating Spectrograms to Square Image Dimensions
            # We use native resolution.
            mels.append(mel)

        # Stack into 3 channels: (Batch, 3, Freq, Time)
        # Requires all mels to have same shape (guaranteed by fixed HOP_LENGTH and N_MELS)
        features = torch.stack(mels, dim=1)

        # ---------------------------------------------------------------------
        # 3. SpecAugment (Training Only)
        # ---------------------------------------------------------------------
        if self.training:
            # Apply masking to the feature map
            features = self.time_mask(features)
            features = self.freq_mask(features)

        # ---------------------------------------------------------------------
        # 4. Normalization (Instance Norm)
        # ---------------------------------------------------------------------
        # Standardize features per sample to help convergence
        mean = features.mean(dim=(2, 3), keepdim=True)
        std = features.std(dim=(2, 3), keepdim=True)
        features = (features - mean) / (std + 1e-6)

        return features
