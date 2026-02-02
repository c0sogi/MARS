import torch
import torch.nn as nn
import numpy as np
from library.config import NOISE_PROB, NOISE_SNR_MIN, NOISE_SNR_MAX, SAMPLE_RATE


class GPUBackgroundNoiseMixer(nn.Module):
    """
    A GPU-resident module for mixing background noise into waveforms.

    This module stores all background noise data in a single continuous GPU tensor
    to allow for fully vectorized slicing and mixing during training.
    """

    def __init__(self, background_noise_list, device=None):
        """
        Args:
            background_noise_list (list of np.ndarray): List of raw audio arrays for background noise.
            device (torch.device, optional): The device to store the noise buffer on.
                                             If None, defaults to 'cuda' if available.
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Filter out extremely short noise clips (shorter than 1 second/16000 samples)
        # to ensure we can always extract a full window.
        valid_noise = [n for n in background_noise_list if len(n) >= SAMPLE_RATE]

        if not valid_noise:
            print(
                "Warning: No valid background noise clips found (all too short). Noise augmentation disabled."
            )
            self.noise_buffer = None
        else:
            # Concatenate all noise clips into one long buffer
            # This enables vectorized indexing later
            full_noise = np.concatenate(valid_noise, axis=0)
            self.register_buffer(
                "noise_buffer", torch.from_numpy(full_noise).float().to(self.device)
            )

        self.noise_prob = NOISE_PROB
        self.snr_min = NOISE_SNR_MIN
        self.snr_max = NOISE_SNR_MAX

    def forward(self, waveforms):
        """
        Args:
            waveforms (torch.Tensor): Input batch of waveforms with shape (Batch, Time).

        Returns:
            torch.Tensor: Augmented waveforms.
        """
        # Only apply augmentation during training and if we have noise data
        if not self.training or self.noise_buffer is None:
            return waveforms

        B, T = waveforms.shape

        # Determine which samples in the batch get noise
        # Shape: (B, 1) to broadcast over time
        apply_noise = torch.rand(B, device=self.device) < self.noise_prob

        # Optimization: If no samples need noise, return early
        if not apply_noise.any():
            return waveforms

        # Ensure we don't index out of bounds
        # Max start index is buffer_length - waveform_length
        buffer_len = self.noise_buffer.numel()
        if buffer_len <= T:
            return waveforms  # Safety check

        max_start = buffer_len - T

        # Generate random start indices for noise extraction
        # Shape: (B,)
        random_starts = torch.randint(0, max_start, (B,), device=self.device)

        # Create indexing grid for vectorized gathering
        # Shape: (B, T)
        # indices[b, t] = random_starts[b] + t
        indices = random_starts.unsqueeze(1) + torch.arange(T, device=self.device)

        # Gather noise segments from the buffer
        # Shape: (B, T)
        noise_segments = self.noise_buffer[indices]

        # Calculate RMS (Root Mean Square) energy
        # Add epsilon to prevent division by zero
        eps = 1e-8

        # Signal RMS: (B, 1)
        sig_rms = torch.sqrt(torch.mean(waveforms**2, dim=1, keepdim=True))

        # Noise RMS: (B, 1)
        noise_rms = torch.sqrt(torch.mean(noise_segments**2, dim=1, keepdim=True))

        # Sample random SNR values (in dB)
        # Shape: (B, 1)
        snr_db = torch.empty(B, 1, device=self.device).uniform_(
            self.snr_min, self.snr_max
        )

        # Calculate required noise scaling factor
        # SNR_db = 20 * log10(sig_rms / (scale * noise_rms))
        # scale = (sig_rms / noise_rms) * 10^(-SNR_db / 20)
        target_noise_rms = sig_rms / (10 ** (snr_db / 20))
        scale = target_noise_rms / (noise_rms + eps)

        # Apply mixing only to selected samples
        # We use 'apply_noise' mask (B,) expanded to (B, T)
        mask = apply_noise.unsqueeze(1).expand(B, T)

        # Mix
        # out = signal + scale * noise
        augmented = waveforms.clone()
        augmented[mask] = waveforms[mask] + (scale * noise_segments)[mask]

        # Clamp to valid audio range [-1, 1]
        augmented = torch.clamp(augmented, -1.0, 1.0)

        return augmented
