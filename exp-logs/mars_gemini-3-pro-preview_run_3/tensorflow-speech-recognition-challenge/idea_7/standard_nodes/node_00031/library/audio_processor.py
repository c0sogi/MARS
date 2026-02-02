import os
import torch
import numpy as np
import soundfile as sf
from library.config import Config


def load_audio(filepath: str, target_samples: int = Config.NUM_SAMPLES) -> torch.Tensor:
    """
    Loads an audio file, converts to mono, resamples to 16kHz, and pads/crops
    to a fixed number of samples.
    """
    if filepath.startswith(Config.INPUT_DIR):
        full_path = filepath
    else:
        full_path = os.path.join(Config.INPUT_DIR, filepath)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Audio file not found: {full_path}")

    try:
        # Load audio using soundfile (returns numpy array)
        # sf.read returns (samples, channels) for multi-channel, or (samples,) for mono
        audio_data, sample_rate = sf.read(full_path)
    except Exception as e:
        raise RuntimeError(f"Error loading {full_path}: {e}")

    # Convert to Tensor
    waveform = torch.from_numpy(audio_data).float()

    # Ensure shape is (Channels, Samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, Samples)
    else:
        waveform = waveform.t()  # (Channels, Samples)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != Config.SAMPLE_RATE:
        # Use torch.nn.functional.interpolate
        # Input must be (Batch, Channels, Time) => (1, 1, Time)
        waveform = waveform.unsqueeze(0)
        new_length = int(waveform.shape[-1] * Config.SAMPLE_RATE / sample_rate)
        waveform = torch.nn.functional.interpolate(
            waveform, size=new_length, mode="linear", align_corners=False
        )
        waveform = waveform.squeeze(0)

    # Fix Length (Pad or Crop)
    current_samples = waveform.shape[1]

    if current_samples < target_samples:
        # Pad with zeros at the end
        padding = target_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_samples > target_samples:
        # Truncate
        waveform = waveform[:, :target_samples]

    return waveform


def _create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    """Creates a Mel filterbank matrix."""
    # FFT bin frequencies
    fft_freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    # Mel scale conversion functions
    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(torch.tensor(f_min, dtype=torch.float32))
    mel_max = hz_to_mel(torch.tensor(f_max, dtype=torch.float32))

    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Create filterbank matrix: (n_mels, n_fft // 2 + 1)
    f_bank = torch.zeros(n_mels, n_fft // 2 + 1)

    for i in range(n_mels):
        f_start = hz_points[i]
        f_center = hz_points[i + 1]
        f_end = hz_points[i + 2]

        # Triangular filter
        # Up slope
        mask_up = (fft_freqs >= f_start) & (fft_freqs <= f_center)
        f_bank[i, mask_up] = (fft_freqs[mask_up] - f_start) / (f_center - f_start)

        # Down slope
        mask_down = (fft_freqs >= f_center) & (fft_freqs <= f_end)
        f_bank[i, mask_down] = (f_end - fft_freqs[mask_down]) / (f_end - f_center)

    return f_bank


def _amplitude_to_db(x, top_db=80.0):
    """Converts power spectrogram to decibels."""
    # Avoid log(0)
    x_db = 10.0 * torch.log10(torch.clamp(x, min=1e-10))
    max_val = x_db.max()
    x_db = torch.clamp(x_db, min=max_val - top_db)
    return x_db


def generate_multires_spectrogram(waveform: torch.Tensor) -> np.ndarray:
    """
    Generates a 3-channel Multi-Resolution Log-Mel Spectrogram from a waveform.
    Uses native PyTorch operations to avoid torchaudio binary dependencies.
    """
    waveform = waveform.cpu()
    # Ensure waveform is 1D (Time,) for STFT processing
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    specs = []

    for res in Config.RESOLUTIONS:
        win_length = res["win_length"]
        n_fft = res["n_fft"]

        # 1. Compute STFT
        window = torch.hann_window(win_length)
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=Config.HOP_LENGTH,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        # 2. Power Spectrogram
        power_spec = stft.abs().pow(2.0)

        # 3. Apply Mel Filterbank
        mel_filters = _create_mel_filterbank(
            Config.SAMPLE_RATE, n_fft, Config.N_MELS, Config.F_MIN, Config.F_MAX
        )

        # Matrix multiplication: (n_mels, freq_bins) @ (freq_bins, time)
        mel_spec = torch.matmul(mel_filters, power_spec)

        # 4. Log Scale (Amplitude to DB)
        log_mel_spec = _amplitude_to_db(mel_spec, top_db=80.0)

        specs.append(log_mel_spec)

    # Stack channels: (3, n_mels, time)
    multires_spec = torch.stack(specs, dim=0)

    return multires_spec.numpy()
