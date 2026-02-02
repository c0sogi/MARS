import torch
import torch.nn as nn
from library import config


class SpecAugment(nn.Module):
    """
    SpecAugment implementation adapted for 3-channel Multi-Resolution Spectrograms.

    This module applies Frequency and Time masking. Crucially, it applies the
    same mask coordinates across all channels (dimension 0) to preserve the
    temporal and frequency alignment of the multi-resolution features.

    Attributes:
        freq_mask_param (int): Maximum possible length of the frequency mask.
        time_mask_param (int): Maximum possible length of the time mask.
    """

    def __init__(
        self,
        freq_mask_param=config.FREQ_MASK_PARAM,
        time_mask_param=config.TIME_MASK_PARAM,
    ):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def frequency_masking(self, spec, min_val):
        """
        Applies frequency masking to the spectrogram.

        Args:
            spec (Tensor): Input spectrogram of shape (C, F, T).
            min_val (float): The value to fill the masked region with.

        Returns:
            Tensor: Frequency masked spectrogram.
        """
        C, F, T = spec.shape

        # Select mask length f from [0, freq_mask_param)
        # Using exclusive upper bound ensures we stay within expected limits
        f = int(torch.randint(0, self.freq_mask_param, (1,)).item())

        # If mask length is 0 or exceeds dimensions, skip
        if f == 0 or f >= F:
            return spec

        # Select start frequency f0 from [0, F - f)
        f0 = int(torch.randint(0, F - f, (1,)).item())

        # Apply mask to all channels simultaneously
        spec[:, f0 : f0 + f, :] = min_val

        return spec

    def time_masking(self, spec, min_val):
        """
        Applies time masking to the spectrogram.

        Args:
            spec (Tensor): Input spectrogram of shape (C, F, T).
            min_val (float): The value to fill the masked region with.

        Returns:
            Tensor: Time masked spectrogram.
        """
        C, F, T = spec.shape

        # Select mask length t from [0, time_mask_param)
        # config.TIME_MASK_PARAM is 20. randint(0, 20) gives max 19.
        # 19/100 time steps is < 20%, satisfying the constraint.
        t = int(torch.randint(0, self.time_mask_param, (1,)).item())

        # If mask length is 0 or exceeds dimensions, skip
        if t == 0 or t >= T:
            return spec

        # Select start time t0 from [0, T - t)
        t0 = int(torch.randint(0, T - t, (1,)).item())

        # Apply mask to all channels simultaneously
        spec[:, :, t0 : t0 + t] = min_val

        return spec

    def forward(self, spec):
        """
        Applies SpecAugment to the input spectrogram.

        Args:
            spec (Tensor): Input tensor of shape (C, F, T).

        Returns:
            Tensor: Augmented tensor.
        """
        # Clone to avoid in-place modification of cached data
        spec = spec.clone()

        # Calculate the minimum value for filling masked regions
        # This is done per-sample to adapt to the dynamic range of the specific clip
        min_val = spec.min()

        # Apply masks
        spec = self.frequency_masking(spec, min_val)
        spec = self.time_masking(spec, min_val)

        return spec
