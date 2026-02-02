import torch
import numpy as np
import random
from library.config import (
    SPEC_AUG_FREQ_MASK_PARAM,
    SPEC_AUG_TIME_MASK_PARAM,
    SPEC_AUG_TIME_MASK_LIMIT,
    RAW_AUG_GAIN_MIN,
    RAW_AUG_GAIN_MAX,
    RAW_AUG_NOISE_SCALE,
)


class SpecAugment:
    """
    Applies SpecAugment (Frequency and Time Masking) to a spectrogram.
    Designed for inputs of shape (..., Freq, Time).
    """

    def __init__(
        self,
        freq_mask_param=SPEC_AUG_FREQ_MASK_PARAM,
        time_mask_param=SPEC_AUG_TIME_MASK_PARAM,
        time_mask_limit_ratio=SPEC_AUG_TIME_MASK_LIMIT,
        p=0.5,
    ):
        """
        Args:
            freq_mask_param (int): Maximum width of the frequency mask.
            time_mask_param (int): Maximum width of the time mask.
            time_mask_limit_ratio (float): Maximum fraction of the time dimension that can be masked.
            p (float): Probability of applying the augmentation.
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.time_mask_limit_ratio = time_mask_limit_ratio
        self.p = p

    def __call__(self, spec):
        """
        Args:
            spec (torch.Tensor): Spectrogram tensor of shape (..., F, T).

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        # Stochastic application
        if random.random() > self.p:
            return spec

        # Clone to avoid in-place modification of the original data
        augmented = spec.clone()

        # Get dimensions
        # Assumes the last two dimensions are Frequency and Time
        ndim = augmented.dim()
        if ndim < 2:
            raise ValueError(
                "Input tensor must have at least 2 dimensions (..., F, T)."
            )

        F = augmented.size(-2)
        T = augmented.size(-1)

        # Fill value is the minimum value in the tensor (e.g., silence/background floor)
        fill_value = augmented.min()

        # --- Frequency Masking ---
        # Draw mask width f from [0, freq_mask_param] (inclusive)
        f = np.random.randint(0, self.freq_mask_param + 1)
        # Clamp f to F
        f = min(f, F)

        if f > 0:
            # Draw start position f0 from [0, F - f] (inclusive)
            f0 = np.random.randint(0, F - f + 1)

            # Create slice for the frequency dimension
            slices = [slice(None)] * ndim
            slices[-2] = slice(f0, f0 + f)

            # Apply mask
            augmented[tuple(slices)] = fill_value

        # --- Time Masking ---
        # Calculate maximum allowed time mask width based on the limit ratio
        max_time_mask = int(T * self.time_mask_limit_ratio)
        # Effective parameter is the minimum of the absolute param and the relative limit
        effective_time_param = min(self.time_mask_param, max_time_mask)

        if effective_time_param > 0:
            # Draw mask width t from [0, effective_time_param] (inclusive)
            t = np.random.randint(0, effective_time_param + 1)
        else:
            t = 0

        # Clamp t to T
        t = min(t, T)

        if t > 0:
            # Draw start position t0 from [0, T - t] (inclusive)
            t0 = np.random.randint(0, T - t + 1)

            # Create slice for the time dimension
            slices = [slice(None)] * ndim
            slices[-1] = slice(t0, t0 + t)

            # Apply mask
            augmented[tuple(slices)] = fill_value

        return augmented


class RawAudioAugment:
    """
    Applies Gain and Gaussian Noise augmentation to raw audio waveforms.
    """

    def __init__(
        self,
        gain_min=RAW_AUG_GAIN_MIN,
        gain_max=RAW_AUG_GAIN_MAX,
        noise_scale=RAW_AUG_NOISE_SCALE,
        p=0.5,
    ):
        """
        Args:
            gain_min (float): Minimum gain factor.
            gain_max (float): Maximum gain factor.
            noise_scale (float): Scale (std dev) of the Gaussian noise.
            p (float): Probability of applying the augmentation.
        """
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.noise_scale = noise_scale
        self.p = p

    def __call__(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Raw audio tensor of shape (..., Time).

        Returns:
            torch.Tensor: Augmented audio tensor.
        """
        if random.random() > self.p:
            return waveform

        # Clone to avoid in-place modification
        augmented = waveform.clone()

        # --- Gain Augmentation ---
        # Sample gain factor uniformly
        gain = random.uniform(self.gain_min, self.gain_max)
        augmented = augmented * gain

        # --- Gaussian Noise Injection ---
        if self.noise_scale > 0:
            # Generate noise with the same shape as input
            noise = torch.randn_like(augmented) * self.noise_scale
            augmented = augmented + noise

        return augmented
