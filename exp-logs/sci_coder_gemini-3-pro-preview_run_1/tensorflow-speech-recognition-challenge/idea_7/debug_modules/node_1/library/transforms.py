import torch
import torch.nn as nn
import torchaudio
from library.config import audio_config, train_config


def get_spectrogram_transform():
    """
    Creates a neural network module that converts raw audio waveforms
    into Log-Mel Spectrograms.

    Returns:
        nn.Sequential: A sequence of transforms (MelSpectrogram -> AmplitudeToDB).
    """
    # Create MelSpectrogram transform using configuration
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_config.sample_rate,
        n_fft=audio_config.n_fft,
        hop_length=audio_config.hop_length,
        n_mels=audio_config.n_mels,
        f_min=audio_config.f_min,
        f_max=audio_config.f_max,
        power=2.0,  # Power spectrogram
    )

    # Convert power spectrogram to decibels
    # top_db=80 is a standard value for audio processing
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    return nn.Sequential(mel_spectrogram, amplitude_to_db)


def get_augmentations():
    """
    Creates a neural network module for SpecAugment data augmentation.
    Includes Time Masking and Frequency Masking.

    Returns:
        nn.Sequential: A sequence of masking transforms.
    """
    # Time Masking: Masks a random part of the time axis
    time_masking = torchaudio.transforms.TimeMasking(
        time_mask_param=train_config.spec_augment_time_mask
    )

    # Frequency Masking: Masks a random part of the frequency axis
    freq_masking = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=train_config.spec_augment_freq_mask
    )

    return nn.Sequential(time_masking, freq_masking)
