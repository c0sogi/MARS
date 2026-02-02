import os
import hashlib
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from library.config import Config

# Ensure deterministic behavior
torch.manual_seed(Config.SEED)


def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """
    Creates a Mel filterbank matrix.
    """
    # FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # Mel points
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mels / 2595) - 1)

    # Bins
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for f in range(left, center):
            filters[i - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            filters[i - 1, f] = (right - f) / (right - center)

    return filters


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self, sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Mel basis
        mel_basis = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Window
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, x):
        # x: (Batch, Time) or (1, Time)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # STFT
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Power spectrogram
        spec = spec.abs().pow(2.0)

        # Mel spectrogram: (Batch, n_mels, Time)
        melspec = torch.matmul(self.mel_basis, spec)
        return melspec


class AmplitudeToDB(torch.nn.Module):
    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        # x is power spectrogram
        x_db = 10 * torch.log10(torch.clamp(x, min=1e-10))
        max_val = x_db.max()
        x_db = torch.clamp(x_db, min=max_val - self.top_db)
        return x_db


def get_audio_transforms():
    """
    Creates and returns the list of MelSpectrogram transforms and the AmplitudeToDB transform
    based on the configuration.
    """
    transforms = []
    # Create a transform for each window length (resolution)
    for win_len, n_fft in zip(Config.WIN_LENGTHS, Config.N_FFTS):
        melspec = MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        transforms.append(melspec)

    # Standard Log-scaling
    to_db = AmplitudeToDB(top_db=80)

    return transforms, to_db


def load_and_process_audio(filepath):
    """
    Loads an audio file, resamples it to the target sample rate, converts to mono,
    and pads/crops it to the fixed duration specified in Config.

    Args:
        filepath (str): Relative path to the audio file.

    Returns:
        torch.Tensor: Processed waveform of shape (1, NUM_SAMPLES).
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)

    # Handle missing files gracefully
    if not os.path.exists(full_path):
        return torch.zeros(1, Config.NUM_SAMPLES)

    try:
        # sf.read returns (samples, channels) or (samples,)
        waveform, sample_rate = sf.read(full_path)
    except Exception:
        # Return silence on load failure
        return torch.zeros(1, Config.NUM_SAMPLES)

    # Convert to torch
    waveform = torch.from_numpy(waveform).float()

    # Handle channels: Ensure (Channels, Time) format for processing
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, Time)
    else:
        waveform = waveform.t()  # (Time, Channels) -> (Channels, Time)

    # Resample if necessary
    if sample_rate != Config.SAMPLE_RATE:
        # Use simple interpolation
        waveform = waveform.unsqueeze(0)  # (1, C, T)
        new_len = int(waveform.shape[-1] * Config.SAMPLE_RATE / sample_rate)
        waveform = torch.nn.functional.interpolate(
            waveform, size=new_len, mode="linear", align_corners=False
        )
        waveform = waveform.squeeze(0)  # (C, T)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed length (Center alignment)
    current_len = waveform.shape[1]
    target_len = Config.NUM_SAMPLES

    if current_len < target_len:
        # Pad with zeros
        pad_amount = target_len - current_len
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        waveform = torch.nn.functional.pad(
            waveform, (pad_left, pad_right), mode="constant", value=0
        )
    elif current_len > target_len:
        # Center crop
        start = (current_len - target_len) // 2
        waveform = waveform[:, start : start + target_len]

    return waveform


def compute_multires_spectrogram(waveform, transforms=None, to_db=None):
    """
    Generates a 3-channel Multi-Resolution Log-Mel Spectrogram from the waveform.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        transforms (list, optional): List of MelSpectrogram transforms.
        to_db (callable, optional): AmplitudeToDB transform.

    Returns:
        torch.Tensor: 3-channel tensor of shape (3, N_MELS, TIME_STEPS).
    """
    if transforms is None or to_db is None:
        transforms, to_db = get_audio_transforms()

    specs = []
    for t in transforms:
        # Compute Mel Spectrogram: (1, n_mels, time)
        spec = t(waveform)
        # Convert to Log Scale
        spec = to_db(spec)
        specs.append(spec)

    # Stack along the channel dimension -> (3, n_mels, time)
    multi_res_spec = torch.cat(specs, dim=0)

    # Ensure strict time dimension consistency
    # For 16000 samples and 160 hop, we expect ~101 frames.
    # We fix it to 101 to handle minor padding differences.
    target_frames = 101
    current_frames = multi_res_spec.shape[2]

    if current_frames > target_frames:
        multi_res_spec = multi_res_spec[:, :, :target_frames]
    elif current_frames < target_frames:
        pad = target_frames - current_frames
        multi_res_spec = torch.nn.functional.pad(multi_res_spec, (0, pad))

    return multi_res_spec


def cache_dataset(df, cache_dir, load_cached_data=True, expand_silence=False):
    """
    Iterates through the metadata DataFrame, processes audio files into Multi-Resolution
    Spectrograms, and saves them as .npy files.

    If expand_silence is True, long 'silence' files are sliced into multiple overlapping
    segments to increase data diversity (Cite solution_lesson_node_00001).

    Args:
        df (pd.DataFrame): Metadata DataFrame containing 'filepath'.
        cache_dir (str): Directory to save cached files.
        load_cached_data (bool): If True, skips processing for existing files.
        expand_silence (bool): Whether to generate multiple clips from silence files.

    Returns:
        pd.DataFrame: A new DataFrame with 'cache_path' column and potentially more rows.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize transforms once to avoid overhead
    transforms, to_db = get_audio_transforms()

    new_rows = []

    # Iterate through metadata
    for _, row in df.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        # Base hash for the file
        file_hash_base = hashlib.md5(filepath.encode("utf-8")).hexdigest()

        # --- Silence Expansion Logic ---
        if expand_silence and label == "silence":
            full_path = os.path.join(Config.INPUT_DIR, filepath)
            if not os.path.exists(full_path):
                continue

            try:
                # Load full audio
                wav, sr = sf.read(full_path)
                wav = torch.from_numpy(wav).float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                else:
                    wav = wav.t()  # (C, T)

                # Resample if needed
                if sr != Config.SAMPLE_RATE:
                    wav = wav.unsqueeze(0)
                    new_len = int(wav.shape[-1] * Config.SAMPLE_RATE / sr)
                    wav = torch.nn.functional.interpolate(
                        wav, size=new_len, mode="linear", align_corners=False
                    )
                    wav = wav.squeeze(0)

                # Mono
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)

                # Slice into overlapping segments
                total_samples = wav.shape[1]
                win_size = Config.NUM_SAMPLES
                stride = win_size // 2  # 50% overlap

                # If file is shorter than window, pad it once and treat as 1 clip
                if total_samples < win_size:
                    pad_amount = win_size - total_samples
                    pad_left = pad_amount // 2
                    pad_right = pad_amount - pad_left
                    chunk = torch.nn.functional.pad(wav, (pad_left, pad_right), value=0)

                    save_path = os.path.join(cache_dir, f"{file_hash_base}_0.npy")
                    if not (load_cached_data and os.path.exists(save_path)):
                        spec = compute_multires_spectrogram(chunk, transforms, to_db)
                        np.save(save_path, spec.numpy())

                    r = row.to_dict()
                    r["cache_path"] = save_path
                    new_rows.append(r)
                else:
                    # Generate multiple clips
                    num_clips = (total_samples - win_size) // stride + 1
                    for i in range(num_clips):
                        start = i * stride
                        end = start + win_size
                        chunk = wav[:, start:end]

                        save_path = os.path.join(cache_dir, f"{file_hash_base}_{i}.npy")

                        if not (load_cached_data and os.path.exists(save_path)):
                            spec = compute_multires_spectrogram(
                                chunk, transforms, to_db
                            )
                            np.save(save_path, spec.numpy())

                        r = row.to_dict()
                        r["cache_path"] = save_path
                        new_rows.append(r)

            except Exception as e:
                print(f"Error processing silence file {filepath}: {e}")
                continue

        else:
            # --- Standard Logic ---
            save_path = os.path.join(cache_dir, f"{file_hash_base}.npy")

            if not (load_cached_data and os.path.exists(save_path)):
                waveform = load_and_process_audio(filepath)
                spec_tensor = compute_multires_spectrogram(waveform, transforms, to_db)
                np.save(save_path, spec_tensor.numpy())

            r = row.to_dict()
            r["cache_path"] = save_path
            new_rows.append(r)

    return pd.DataFrame(new_rows)
