import torch
import torch.nn as nn
import torchaudio
from library.config import Config


class InstanceNormalization(nn.Module):
    """
    Applies Instance Normalization to the input spectrogram.
    Standardizes the signal volume and contrast per sample by calculating
    statistics over the frequency and time dimensions.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (..., Freq, Time).
        Returns:
            Tensor: Normalized tensor of the same shape.
        """
        # Calculate mean and std over the last two dimensions (Freq, Time)
        # keepdim=True ensures broadcasting works correctly
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)

        return (x - mean) / (std + self.eps)


def get_transforms(phase: str = "train"):
    """
    Constructs the audio transformation pipeline.

    Args:
        phase (str): The current phase ('train', 'val', or 'test').
                     Augmentations are only applied if phase == 'train'.

    Returns:
        nn.Sequential: A composed sequence of transforms.
    """
    transforms = []

    # 1. MelSpectrogram
    # Configured for high temporal resolution (25ms window, 10ms hop)
    # Input: (..., Time) -> Output: (..., n_mels, Time)
    transforms.append(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,  # Power spectrogram
        )
    )

    # 2. AmplitudeToDB
    # Converts power spectrogram to decibel scale
    transforms.append(torchaudio.transforms.AmplitudeToDB(stype="power"))

    # 3. Instance Normalization
    # Standardize per sample
    transforms.append(InstanceNormalization())

    # 4. SpecAugment (Training Only)
    if phase == "train":
        # Frequency Masking
        transforms.append(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            )
        )
        # Time Masking
        transforms.append(
            torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        )

    return nn.Sequential(*transforms)
