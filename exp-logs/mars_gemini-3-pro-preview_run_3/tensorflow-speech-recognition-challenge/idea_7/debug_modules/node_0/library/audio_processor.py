import os
import torch
import torchaudio
import numpy as np
from library.config import Config


def load_audio(filepath: str, target_samples: int = Config.NUM_SAMPLES) -> torch.Tensor:
    """
    Loads an audio file, converts to mono, resamples to 16kHz, and pads/crops
    to a fixed number of samples.

    Args:
        filepath (str): Relative path to the audio file (e.g., from metadata).
        target_samples (int): The fixed number of samples required (default: 16000).

    Returns:
        torch.Tensor: A tensor of shape (1, target_samples).
    """
    # Construct full path. Metadata paths are relative to input root.
    # Check if filepath already contains the input dir to be safe.
    if filepath.startswith(Config.INPUT_DIR):
        full_path = filepath
    else:
        full_path = os.path.join(Config.INPUT_DIR, filepath)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Audio file not found: {full_path}")

    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(full_path)
    except Exception as e:
        raise RuntimeError(f"Error loading {full_path}: {e}")

    # Convert to Mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Fix Length (Pad or Crop)
    current_samples = waveform.shape[1]

    if current_samples < target_samples:
        # Pad with zeros at the end
        padding = target_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_samples > target_samples:
        # Truncate (Take the first target_samples)
        # For command recognition, the keyword is usually at the start or centered.
        # Deterministic truncation is preferred for the processor module.
        waveform = waveform[:, :target_samples]

    return waveform


def generate_multires_spectrogram(waveform: torch.Tensor) -> np.ndarray:
    """
    Generates a 3-channel Multi-Resolution Log-Mel Spectrogram from a waveform.

    Channels correspond to different STFT window sizes (Short, Medium, Long)
    to capture different time-frequency trade-offs.

    Args:
        waveform (torch.Tensor): Input waveform of shape (1, samples).

    Returns:
        numpy.ndarray: 3-channel spectrogram of shape (3, n_mels, time_steps).
    """
    # Ensure waveform is on CPU for transforms to avoid unnecessary GPU VRAM usage during data loading
    waveform = waveform.cpu()

    specs = []

    for res in Config.RESOLUTIONS:
        # Extract resolution-specific parameters
        win_length = res["win_length"]
        n_fft = res["n_fft"]

        # Configure MelSpectrogram Transform
        # center=True is crucial to align time steps across different window sizes
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        )

        # Configure AmplitudeToDB Transform
        # top_db=80.0 clamps the dynamic range to 80dB, standard for audio tasks
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        # Compute Features
        # 1. Mel Spectrogram
        mel_spec = mel_transform(waveform)  # Shape: (1, n_mels, time)

        # 2. Log Scale
        log_mel_spec = db_transform(mel_spec)

        specs.append(log_mel_spec)

    # Stack the 3 resolutions along the channel dimension (dim=0)
    # Result shape: (3, n_mels, time)
    multires_spec = torch.cat(specs, dim=0)

    return multires_spec.numpy()
