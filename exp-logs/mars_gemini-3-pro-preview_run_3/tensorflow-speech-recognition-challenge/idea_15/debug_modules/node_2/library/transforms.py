import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import numpy as np
from library.config import Config


class GPUAudioPreprocess(nn.Module):
    """
    GPU-Native Audio Preprocessing Module.
    Handles Physics-Based Augmentations (Pitch Shift, Time Stretch),
    Multi-Resolution Spectrogram generation, and SpecAugment.
    """

    def __init__(self, device=None):
        super().__init__()
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ==========================================
        # Spectrogram Transforms (Multi-Resolution)
        # ==========================================
        # Short Window (20ms) - High Temporal Resolution
        self.mel_short = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT_SHORT,
            win_length=Config.WIN_LENGTH_SHORT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
        )

        # Medium Window (40ms) - Balanced
        self.mel_medium = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT_MEDIUM,
            win_length=Config.WIN_LENGTH_MEDIUM,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
        )

        # Long Window (60ms) - High Frequency Resolution
        self.mel_long = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT_LONG,
            win_length=Config.WIN_LENGTH_LONG,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # ==========================================
        # SpecAugment
        # ==========================================
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.MASK_TIME_LIMIT
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.MASK_FREQ_LIMIT
        )

    def random_pitch_shift(self, waveform):
        """
        Applies random pitch shifting to the waveform.
        """
        # Select a random step within the range
        steps = random.uniform(-Config.PITCH_SHIFT_STEPS, Config.PITCH_SHIFT_STEPS)
        # Apply pitch shift (works on GPU)
        return torchaudio.functional.pitch_shift(waveform, Config.SAMPLE_RATE, steps)

    def random_time_stretch(self, waveform):
        """
        Applies random time stretching via resampling (Speed Perturbation).
        """
        # Select random rate
        rate = random.uniform(1.0 / Config.TIME_STRETCH_RATE, Config.TIME_STRETCH_RATE)

        # Calculate new resampling target frequency to simulate speed change
        # If we want to play faster (rate > 1), we resample to fewer samples.
        new_freq = int(Config.SAMPLE_RATE * rate)

        # Resample
        return torchaudio.functional.resample(waveform, Config.SAMPLE_RATE, new_freq)

    def add_noise(self, waveform, noise):
        """
        Mixes background noise into the waveform.
        """
        # Ensure noise matches waveform length
        if noise.shape[-1] < waveform.shape[-1]:
            # Pad noise by repeating or zero-padding
            padding = waveform.shape[-1] - noise.shape[-1]
            noise = F.pad(noise, (0, padding))
        elif noise.shape[-1] > waveform.shape[-1]:
            # Random crop of noise
            start = random.randint(0, noise.shape[-1] - waveform.shape[-1])
            noise = noise[..., start : start + waveform.shape[-1]]

        # Random SNR between 10 and 30 dB
        snr_db = random.uniform(10, 30)

        # Calculate powers
        sig_power = waveform.pow(2).mean(dim=-1, keepdim=True)
        noise_power = noise.pow(2).mean(dim=-1, keepdim=True)

        # Calculate scaling factor
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_power / (noise_power + 1e-9))

        return waveform + noise * scale

    def forward(self, x, training=False, noise=None):
        """
        Forward pass for the preprocessing pipeline.

        Args:
            x (Tensor): Input waveforms (Batch, Time).
            training (bool): Whether to apply augmentations.
            noise (Tensor, optional): Background noise waveforms for mixing.

        Returns:
            Tensor: Preprocessed 3-channel spectrograms (Batch, 3, 224, 224).
        """
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)

        # ==========================================
        # 1. Waveform Augmentations
        # ==========================================
        if training:
            # Random Pitch Shift
            if random.random() < 0.5:
                x = self.random_pitch_shift(x)

            # Random Time Stretch
            if random.random() < 0.5:
                x = self.random_time_stretch(x)

            # Background Noise Injection
            if noise is not None and random.random() < Config.NOISE_PROB:
                if noise.device != self.device:
                    noise = noise.to(self.device)
                x = self.add_noise(x, noise)

        # ==========================================
        # 2. Length Normalization (Crop/Pad)
        # ==========================================
        target_len = int(Config.SAMPLE_RATE * Config.DURATION)
        current_len = x.shape[-1]

        if current_len > target_len:
            if training:
                # Random crop
                start = random.randint(0, current_len - target_len)
                x = x[..., start : start + target_len]
            else:
                # Center crop
                start = (current_len - target_len) // 2
                x = x[..., start : start + target_len]
        elif current_len < target_len:
            # Pad with zeros
            pad_amt = target_len - current_len
            x = F.pad(x, (0, pad_amt))

        # ==========================================
        # 3. Multi-Resolution Spectrogram Generation
        # ==========================================
        # Generate Log-Mels
        mels_s = self.amplitude_to_db(self.mel_short(x))
        mels_m = self.amplitude_to_db(self.mel_medium(x))
        mels_l = self.amplitude_to_db(self.mel_long(x))

        # Resize to Target Image Size (224, 224)
        # Input to interpolate: (Batch, Channel, Height, Width)
        # Mel Output: (Batch, Freq, Time) -> Treat as (Batch, 1, Freq, Time)
        def resize_mel(m):
            m = m.unsqueeze(1)
            # align_corners=False is standard for image resizing
            m = F.interpolate(
                m, size=Config.TARGET_IMAGE_SIZE, mode="bilinear", align_corners=False
            )
            return m.squeeze(1)

        mels_s = resize_mel(mels_s)
        mels_m = resize_mel(mels_m)
        mels_l = resize_mel(mels_l)

        # Stack into 3 Channels: (Batch, 3, 224, 224)
        image = torch.stack([mels_s, mels_m, mels_l], dim=1)

        # ==========================================
        # 4. SpecAugment (Spectrogram Level)
        # ==========================================
        if training:
            # Apply masking. We flatten channels to apply mask to each channel independently
            # or we can view it as a batch of B*3 images.
            B, C, H, W = image.shape
            image_flat = image.view(B * C, H, W)

            if random.random() < Config.MASK_TIME_PROB:
                image_flat = self.time_masking(image_flat)

            if random.random() < Config.MASK_FREQ_PROB:
                image_flat = self.freq_masking(image_flat)

            image = image_flat.view(B, C, H, W)

        # ==========================================
        # 5. Normalization (Instance Standardization)
        # ==========================================
        # Normalize per sample to mean 0, std 1
        mean = image.mean(dim=(2, 3), keepdim=True)
        std = image.std(dim=(2, 3), keepdim=True)
        image = (image - mean) / (std + 1e-6)

        return image
