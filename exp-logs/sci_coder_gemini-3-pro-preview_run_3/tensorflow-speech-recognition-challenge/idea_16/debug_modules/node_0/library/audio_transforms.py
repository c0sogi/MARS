import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from library.config import Config


class GPUAudioProcessor(nn.Module):
    """
    A GPU-native audio processing module that handles:
    1. Physics-based Augmentation (Pitch Shift, Time Stretch/Resampling, Noise Mixing) on raw waveforms.
    2. Multi-Resolution Log-Mel Spectrogram extraction.
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
        # We create a list of MelSpectrogram transforms for different window sizes.
        # This maps time-frequency uncertainty trade-offs to RGB channels.
        self.mel_transforms = nn.ModuleList()
        for n_fft, hop_length in zip(Config.N_FFT_LIST, Config.HOP_LENGTH_LIST):
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                win_length=n_fft,  # Use n_fft as window length
                hop_length=hop_length,
                n_mels=Config.N_MELS,
                f_min=Config.F_MIN,
                f_max=Config.F_MAX,
                power=2.0,
                normalized=True,
            )
            self.mel_transforms.append(transform)

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        # ---------------------------------------------------------------------
        # 2. SpecAugment Transforms
        # ---------------------------------------------------------------------
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.MASK_FREQ_LIMIT
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.MASK_TIME_LIMIT, p=Config.MASK_TIME_PROB
        )

        # ---------------------------------------------------------------------
        # 3. Background Noise Loading
        # ---------------------------------------------------------------------
        self.noises = []
        if os.path.exists(Config.NOISE_DIR):
            noise_files = [
                f for f in os.listdir(Config.NOISE_DIR) if f.endswith(".wav")
            ]
            for nf in noise_files:
                path = os.path.join(Config.NOISE_DIR, nf)
                try:
                    # Load and move to GPU immediately
                    waveform, sr = torchaudio.load(path)
                    if sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(
                            waveform, sr, self.sample_rate
                        )

                    # Store as (Channels, Time) on GPU
                    self.noises.append(waveform.to(self.device))
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
                noise = random.choice(self.noises)  # (C, T)
                noise_len = noise.shape[1]

                # Get a random crop of the noise
                if noise_len <= self.num_samples:
                    repeats = math.ceil(self.num_samples / noise_len)
                    noise_crop = noise.repeat(1, repeats)[..., : self.num_samples]
                else:
                    start = torch.randint(
                        0, noise_len - self.num_samples, (1,), device=self.device
                    )
                    noise_crop = noise[..., start : start + self.num_samples]

                # Calculate scaling for random SNR
                sig_rms = augmented[i].norm(p=2)
                noise_rms = noise_crop.norm(p=2)

                if noise_rms > 1e-6:
                    # SNR between 5 and 20 dB
                    snr_db = torch.empty(1, device=self.device).uniform_(5, 20)
                    scale = (sig_rms / noise_rms) * torch.pow(10, -snr_db / 20)

                    # Add noise (squeeze channel dim of noise)
                    augmented[i] = augmented[i] + (scale * noise_crop.squeeze(0))

        return augmented

    def _apply_physics_aug(self, waveforms):
        """
        Applies Pitch Shift and Time Stretch (Resampling).
        """
        augmented = waveforms.clone()

        # 1. Random Pitch Shift
        if torch.rand(1, device=self.device) < self.aug_prob:
            # Shift by random semitones [-2, 2]
            n_steps = (
                torch.empty(1, device=self.device)
                .uniform_(-Config.PITCH_SHIFT_SEMITONES, Config.PITCH_SHIFT_SEMITONES)
                .item()
            )
            try:
                augmented = torchaudio.functional.pitch_shift(
                    augmented, self.sample_rate, n_steps=n_steps
                )
            except:
                pass  # Fallback if backend issues

        # 2. Time Stretch (Speed Perturbation via Resampling)
        if torch.rand(1, device=self.device) < self.aug_prob:
            # Speed factor: 0.9 (slow) to 1.1 (fast)
            speed = (
                torch.empty(1, device=self.device)
                .uniform_(
                    1.0 - Config.TIME_STRETCH_RATE, 1.0 + Config.TIME_STRETCH_RATE
                )
                .item()
            )

            # To play faster (speed > 1), we need fewer samples -> resample to lower freq
            # To play slower (speed < 1), we need more samples -> resample to higher freq
            target_freq = int(self.sample_rate / speed)

            # Resample batch
            resampled = torchaudio.functional.resample(
                augmented, orig_freq=self.sample_rate, new_freq=target_freq
            )

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

            # Resize to ImageNet size (224, 224)
            # Interpolate requires (Batch, Channel, H, W)
            # We treat (Freq, Time) as (H, W)
            mel = mel.unsqueeze(1)  # Add channel dim

            mel = F.interpolate(
                mel, size=Config.IMG_SIZE, mode="bilinear", align_corners=False
            )

            mel = mel.squeeze(1)  # Remove channel dim
            mels.append(mel)

        # Stack into 3 channels: (Batch, 3, 224, 224)
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
