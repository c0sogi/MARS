import os
import hashlib
import numpy as np
import torch
import soundfile as sf
from library.config import Config


class LogMelSpectrogram(torch.nn.Module):
    """
    Native PyTorch implementation of Log-Mel Spectrogram to replace torchaudio.
    """

    def __init__(
        self, sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Create Mel Basis
        mel_basis = self._create_mel_basis(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Create Window (Hann)
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def _create_mel_basis(self, sr, n_fft, n_mels, f_min, f_max):
        """
        Creates a Mel Filterbank matrix (Slaney-like normalization).
        """

        # HTK Mel scale formulas
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        # Points in Mel domain
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Points in FFT bins: bin = freq * n_fft / sr
        bin_points = np.floor(hz_points * n_fft / sr).astype(int)

        # Create Filterbank
        n_freqs = n_fft // 2 + 1
        weights = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            start = bin_points[i]
            center = bin_points[i + 1]
            end = bin_points[i + 2]

            # Upslope
            if center > start:
                weights[i, start:center] = (np.arange(start, center) - start) / (
                    center - start
                )

            # Downslope
            if end > center:
                weights[i, center:end] = (end - np.arange(center, end)) / (end - center)

        # Slaney Area Normalization (approximate)
        # Normalize by the width of the mel band in Hz to maintain energy
        enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
        weights *= enorm[:, np.newaxis]

        return weights

    def forward(self, waveform):
        # waveform: (Batch, Time) or (1, Time)

        # STFT
        complex_spec = torch.stft(
            waveform,
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

        # Power Spectrogram: |STFT|^2
        power_spec = complex_spec.abs().pow(2.0)

        # Mel Spectrogram: Mel_Basis @ Power_Spec
        # Mel_Basis: (n_mels, n_freqs)
        # Power_Spec: (Batch, n_freqs, Time)
        # Result: (Batch, n_mels, Time)
        mel_spec = torch.matmul(self.mel_basis, power_spec)

        # Log Scale (Natural Log)
        eps = 1e-9
        log_mel = torch.log(mel_spec + eps)

        return log_mel


class AudioProcessor:
    """
    Handles the conversion of raw audio into 3-Channel Multi-Resolution Log-Mel Spectrograms.
    Implements caching to speed up training.
    """

    def __init__(self):
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Use CPU for preprocessing to avoid GPU memory contention during data loading
        self.device = torch.device("cpu")

        # Initialize MelSpectrogram transforms for each resolution
        self.transforms = []
        for n_fft, win_len in zip(Config.N_FFT_RESOLUTIONS, Config.WIN_LENGTHS):
            transform = LogMelSpectrogram(
                sample_rate=Config.SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                f_min=Config.F_MIN,
                f_max=Config.F_MAX,
            ).to(self.device)
            self.transforms.append(transform)

    def _load_audio(self, filepath):
        """
        Loads audio from disk using SoundFile, resamples, mixes to mono, and pads/crops.
        """
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        if not os.path.exists(full_path):
            return torch.zeros(1, Config.N_SAMPLES)

        try:
            # soundfile returns (samples, channels) or (samples,)
            data, sr = sf.read(full_path)
            waveform = torch.from_numpy(data).float()

            # Ensure (Channels, Time) format
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, Time)
            else:
                waveform = waveform.t()  # (Channels, Time)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, Config.N_SAMPLES)

        # Resample if necessary
        if sr != Config.SAMPLE_RATE:
            # Interpolate expects (Batch, Channels, Time)
            waveform = waveform.unsqueeze(0)
            new_len = int(waveform.shape[-1] * Config.SAMPLE_RATE / sr)
            waveform = torch.nn.functional.interpolate(
                waveform, size=new_len, mode="linear", align_corners=False
            )
            waveform = waveform.squeeze(0)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure fixed length (Pad or Crop)
        num_samples = waveform.shape[1]
        target_samples = Config.N_SAMPLES

        if num_samples < target_samples:
            # Pad with zeros at the end
            padding = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > target_samples:
            # Crop (simple truncation)
            waveform = waveform[:, :target_samples]

        return waveform

    def _compute_spectrograms(self, waveform):
        """
        Computes the 3-channel Multi-Resolution Log-Mel Spectrogram.
        """
        waveform = waveform.to(self.device)
        specs = []

        for transform in self.transforms:
            # Compute Log Mel Spectrogram: Shape (1, n_mels, time)
            # Our custom transform already applies log
            log_mel = transform(waveform)
            specs.append(log_mel)

        # Stack along channel dimension: Shape (3, n_mels, time)
        multi_res_spec = torch.cat(specs, dim=0)

        # Apply Global Normalization
        multi_res_spec = (multi_res_spec - Config.NORM_MEAN) / Config.NORM_STD

        return multi_res_spec

    def process_file(self, filepath, load_cached_data=True):
        """
        Main processing function with caching mechanism.

        Args:
            filepath (str): Relative path to the audio file.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Processed features of shape (3, n_mels, time).
        """
        # Generate a unique cache key based on the filepath
        file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        cache_filename = f"{file_hash}.npy"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data
            except Exception:
                # If load fails (corrupt file), proceed to recompute
                pass

        # 2. Compute from scratch
        waveform = self._load_audio(filepath)
        features_tensor = self._compute_spectrograms(waveform)

        # Convert to numpy for storage/usage
        features_np = features_tensor.cpu().numpy()

        # 3. Save to cache
        try:
            np.save(cache_path, features_np)
        except Exception as e:
            print(f"Warning: Failed to save cache for {filepath}: {e}")

        return features_np
