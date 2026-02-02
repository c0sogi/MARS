import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import TimeMasking, FrequencyMasking
from library.config import Config


class SpecAugment(nn.Module):
    """
    Applies SpecAugment (Frequency Masking and Time Masking) to spectrograms.
    This module is designed to be part of the preprocessing or model pipeline.
    """

    def __init__(
        self,
        freq_mask_param=None,
        time_mask_param=None,
        num_freq_masks=2,
        num_time_masks=2,
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param (int, optional): Maximum possible length of the frequency mask.
                                             Defaults to ~20% of N_MELS.
            time_mask_param (int, optional): Maximum possible length of the time mask.
                                             Defaults to ~10% of the estimated time steps.
            num_freq_masks (int): Number of frequency masks to apply.
            num_time_masks (int): Number of time masks to apply.
        """
        super().__init__()

        # Determine defaults based on Config if not provided
        if freq_mask_param is None:
            # Mask up to ~20% of frequency bins
            freq_mask_param = int(Config.N_MELS * 0.2)

        if time_mask_param is None:
            # Estimate time steps: (SR * DURATION) / HOP_LENGTH
            # For 5s @ 32kHz with hop 320, steps ~ 500. Mask ~10% -> 50.
            time_steps = int((Config.SR * Config.DURATION) / Config.HOP_LENGTH)
            time_mask_param = int(time_steps * 0.1)

        # Create sequences of masking transforms
        self.freq_masking = nn.Sequential(
            *[
                FrequencyMasking(freq_mask_param=freq_mask_param)
                for _ in range(num_freq_masks)
            ]
        )
        self.time_masking = nn.Sequential(
            *[
                TimeMasking(time_mask_param=time_mask_param)
                for _ in range(num_time_masks)
            ]
        )

    def forward(self, spec):
        """
        Apply SpecAugment to the input spectrogram.

        Args:
            spec (torch.Tensor): Input spectrogram of shape (..., Freq, Time).

        Returns:
            torch.Tensor: Masked spectrogram.
        """
        # torchaudio transforms operate on the last two dimensions (Freq, Time)
        # Apply frequency masking
        spec = self.freq_masking(spec)
        # Apply time masking
        spec = self.time_masking(spec)
        return spec


def mixup_data(data, target, alpha=0.4, device=None):
    """
    Applies Mixup augmentation to the input data and targets.

    This implementation mixes the targets directly, which is suitable for
    BCEWithLogitsLoss in a multi-label setting.

    Args:
        data (torch.Tensor): Input batch of shape (Batch, ...).
        target (torch.Tensor): Target batch of shape (Batch, Num_Classes).
        alpha (float): Mixup hyperparameter.
        device (torch.device, optional): Device to perform operations on.
                                         If None, uses data.device.

    Returns:
        mixed_data (torch.Tensor): The mixed input data.
        mixed_target (torch.Tensor): The mixed target labels.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    if device is None:
        device = data.device

    # Generate random permutation of indices
    index = torch.randperm(batch_size).to(device)

    # Mix inputs
    mixed_data = lam * data + (1 - lam) * data[index, :]

    # Mix targets (linear combination of labels)
    mixed_target = lam * target + (1 - lam) * target[index, :]

    return mixed_data, mixed_target
