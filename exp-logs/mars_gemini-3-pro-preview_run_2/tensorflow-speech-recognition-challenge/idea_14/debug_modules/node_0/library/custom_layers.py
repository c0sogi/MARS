import torch
import torch.nn as nn
import torchaudio
from library.config import Config


class GPUNoiseInjector(nn.Module):
    """
    Mixes background noise into the input waveforms on the GPU.
    Leverages a pre-loaded buffer of background noise for zero-latency augmentation.
    """

    def __init__(self, noise_tensor, min_snr_db=5.0, max_snr_db=30.0, p=0.5):
        super().__init__()
        # Register the noise tensor as a buffer so it moves with the model to GPU
        self.register_buffer("noise_tensor", noise_tensor)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input waveforms of shape (Batch, Time).
        Returns:
            torch.Tensor: Augmented waveforms.
        """
        # Only apply augmentation during training and if probability > 0
        if not self.training or self.p <= 0.0:
            return x

        B, T = x.shape
        device = x.device
        noise_len = self.noise_tensor.size(0)

        # Safety check: If noise buffer is shorter than waveform, skip
        if noise_len < T:
            return x

        # Determine which samples in the batch to augment
        mask = torch.rand(B, device=device) < self.p
        if not mask.any():
            return x

        # 1. Select random noise segments for the batch
        # Generate random start indices for each sample in the batch
        starts = torch.randint(0, noise_len - T + 1, (B,), device=device)

        # Create indices for gathering: (B, T)
        # offsets (1, T) + starts (B, 1) -> (B, T)
        indices = starts.unsqueeze(1) + torch.arange(T, device=device).unsqueeze(0)

        # Gather noise segments efficiently
        noise_segments = self.noise_tensor[indices]  # (B, T)

        # 2. Calculate Signal and Noise Power
        # (B, 1)
        sig_power = x.pow(2).mean(dim=1, keepdim=True)
        noise_power = noise_segments.pow(2).mean(dim=1, keepdim=True)

        # 3. Determine Target SNR
        # Sample SNR from uniform distribution [min, max]
        target_snr_db = (
            torch.rand(B, 1, device=device) * (self.max_snr_db - self.min_snr_db)
            + self.min_snr_db
        )
        target_snr = 10 ** (target_snr_db / 10.0)

        # 4. Calculate Scaling Factor
        # scale = sqrt( P_signal / (P_noise * SNR) )
        # Add epsilon to avoid division by zero
        scale = torch.sqrt(sig_power / (target_snr * noise_power + 1e-9))

        # 5. Mix
        # x_aug = x + scale * noise
        augmented = x + scale * noise_segments

        # 6. Apply Mask
        # Select augmented or original based on the random mask
        mask_expanded = mask.unsqueeze(1)  # (B, 1)
        out = torch.where(mask_expanded, augmented, x)

        # Clip to valid audio range [-1, 1]
        out = torch.clamp(out, -1.0, 1.0)

        return out


class DifferentiableSpectrogram(nn.Module):
    """
    Converts waveforms to Log-Mel Spectrograms with Instance Normalization.
    Operates as a differentiable layer within the model.
    """

    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            n_mels=Config.N_MELS,
            power=2.0,
        )
        # Instance Normalization per channel (we have 1 channel).
        # affine=False ensures we strictly normalize to mean=0, std=1 without learnable parameters.
        self.instance_norm = nn.InstanceNorm2d(1, affine=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Waveforms (Batch, Time)
        Returns:
            torch.Tensor: Normalized Log-Mel Spectrograms (Batch, 1, F, T)
        """
        # 1. Compute Mel Spectrogram
        # Output: (Batch, n_mels, time)
        spec = self.mel_transform(x)

        # 2. Log Transform (Log-Mel)
        # Add epsilon for numerical stability
        spec = torch.log(spec + 1e-9)

        # 3. Add Channel Dimension
        # (Batch, 1, F, T)
        spec = spec.unsqueeze(1)

        # 4. Instance Normalization
        # Normalizes across Frequency and Time dimensions for each sample
        spec = self.instance_norm(spec)

        return spec


class SpecAugment(nn.Module):
    """
    Applies Time and Frequency Masking to spectrograms for regularization.
    """

    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Spectrograms (Batch, C, F, T)
        Returns:
            torch.Tensor: Masked Spectrograms
        """
        if not self.training:
            return x

        # Apply Frequency Masking
        x = self.freq_mask(x)

        # Apply Time Masking
        x = self.time_mask(x)

        return x


class AttentionPooling(nn.Module):
    """
    Single-Head 2D Attention Pooling.
    Aggregates spatial features (F, T) into a global embedding using learned attention weights.
    Acts as a learned Voice Activity Detector (VAD).
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Convolution to compute attention scores from features
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature map (Batch, C, H, W)
        Returns:
            torch.Tensor: Global embedding (Batch, C)
        """
        # 1. Compute Attention Logits
        # (Batch, 1, H, W)
        attn_logits = self.attn_conv(x)

        # 2. Spatial Softmax
        B, _, H, W = attn_logits.shape
        # Flatten spatial dims: (Batch, 1, H*W)
        attn_flat = attn_logits.view(B, 1, -1)
        # Softmax over H*W to get probability distribution
        attn_weights = torch.softmax(attn_flat, dim=-1)
        # Reshape back: (Batch, 1, H, W)
        attn_weights = attn_weights.view(B, 1, H, W)

        # 3. Weighted Sum
        # (Batch, C, H, W) * (Batch, 1, H, W) -> (Batch, C, H, W)
        weighted_features = x * attn_weights

        # Sum over spatial dimensions to get global vector
        # (Batch, C)
        global_embedding = weighted_features.sum(dim=(2, 3))

        return global_embedding
