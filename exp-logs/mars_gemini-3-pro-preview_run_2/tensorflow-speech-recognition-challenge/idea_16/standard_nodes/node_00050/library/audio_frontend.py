import torch
import torch.nn as nn
import torchaudio
from library.config import Config


class DifferentiableFrontend(nn.Module):
    """
    GPU-Resident End-to-End Audio Preprocessing and Augmentation Module.

    This module is designed to be the first layer of the neural network.
    It accepts raw waveforms and outputs normalized Log-Mel Spectrograms,
    performing all operations (including augmentation) on the GPU.

    Pipeline:
    1. Dynamic Background Noise Mixing (Training only)
    2. Mel Spectrogram Extraction
    3. Logarithmic Scaling
    4. Instance Normalization
    5. SpecAugment (Time/Freq Masking) (Training only)
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Spectrogram Configuration
        # ==========================================
        # We use torchaudio's MelSpectrogram which uses STFT under the hood.
        # Parameters are tuned for 16kHz audio to capture fine phonetic details.
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LEN,
            hop_length=Config.HOP_LEN,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            power=2.0,  # Power spectrogram
            normalized=False,  # We apply InstanceNorm manually later
        )

        # ==========================================
        # 2. Augmentation Configuration
        # ==========================================
        # SpecAugment: Masking blocks of frequency or time steps.
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK, iid_masks=True
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK, iid_masks=True
        )

        # ==========================================
        # 3. Normalization
        # ==========================================
        # Instance Normalization aligns the dynamic range of audio features
        # with the expected input statistics of standard vision backbones.
        # We use affine=False to strictly normalize without learnable parameters here.
        self.instance_norm = nn.InstanceNorm2d(1, affine=False)

        # Constant for numerical stability in log
        self.log_offset = 1e-6

    def forward(self, x, noise_bank=None):
        """
        Forward pass of the frontend.

        Args:
            x (torch.Tensor): Raw input waveforms. Shape: (Batch, Time)
            noise_bank (list[torch.Tensor], optional): List of background noise tensors
                                                       resident on GPU for augmentation.

        Returns:
            torch.Tensor: Processed spectrogram features. Shape: (Batch, 1, F, T)
        """
        # Ensure input is on the correct device
        if x.device != Config.DEVICE:
            x = x.to(Config.DEVICE)

        # ==========================================
        # 1. Dynamic Background Noise Mixing
        # ==========================================
        # Only applied during training if a noise bank is provided
        if self.training and noise_bank is not None and len(noise_bank) > 0:
            # Determine which samples to augment based on probability
            batch_size = x.size(0)
            # Generate a mask: True where we should inject noise
            augment_mask = (
                torch.rand(batch_size, device=x.device) < Config.NOISE_INJECTION_PROB
            )

            if augment_mask.any():
                aug_indices = torch.where(augment_mask)[0]

                # Randomly select a noise clip index for each sample to be augmented
                noise_indices = torch.randint(
                    0, len(noise_bank), (len(aug_indices),), device=x.device
                )

                # Iterate through selected samples to mix noise
                # (Loop is efficient here as batch size is small, e.g., 32)
                for i, idx in enumerate(aug_indices):
                    noise_clip = noise_bank[noise_indices[i]]

                    waveform_len = x.size(1)
                    noise_len = noise_clip.size(0)

                    # Prepare the noise segment matching the waveform length
                    if noise_len > waveform_len:
                        # Random crop
                        max_start = noise_len - waveform_len
                        start = torch.randint(
                            0, max_start + 1, (1,), device=x.device
                        ).item()
                        noise_segment = noise_clip[start : start + waveform_len]
                    else:
                        # Tile/Repeat if noise is shorter than waveform
                        repeats = (waveform_len // noise_len) + 1
                        noise_segment = noise_clip.repeat(repeats)[:waveform_len]

                    # Random mixing weight (SNR)
                    # Typically 0.0 to 0.15 provides robustness without destroying the signal
                    noise_gain = torch.rand(1, device=x.device) * 0.15

                    # Add noise
                    x[idx] = x[idx] + noise_segment * noise_gain

                # Clamp values to valid audio range [-1, 1]
                x = torch.clamp(x, -1.0, 1.0)

        # ==========================================
        # 2. Feature Extraction (Log-Mel)
        # ==========================================
        # Generate Mel Spectrogram: (Batch, n_mels, time)
        spec = self.mel_spectrogram(x)

        # Log Transform: Convert power to log scale (dB-like)
        spec = torch.log(spec + self.log_offset)

        # Add Channel Dimension: (Batch, 1, n_mels, time)
        spec = spec.unsqueeze(1)

        # ==========================================
        # 3. Normalization
        # ==========================================
        # Normalize per instance to zero mean and unit variance
        spec = self.instance_norm(spec)

        # ==========================================
        # 4. SpecAugment
        # ==========================================
        # Only applied during training
        if self.training:
            # Mask random frequency bands
            spec = self.freq_masking(spec)
            # Mask random time steps
            spec = self.time_masking(spec)

        return spec
