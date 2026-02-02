import os
import hashlib
import numpy as np
import torch
import torchaudio
from library.config import Config


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
        # We use the resolutions defined in Config to create 3 different views of the audio
        self.transforms = []
        for n_fft, win_len in zip(Config.N_FFT_RESOLUTIONS, Config.WIN_LENGTHS):
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=Config.SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                f_min=Config.F_MIN,
                f_max=Config.F_MAX,
                center=True,  # Ensures time dimension is consistent across resolutions
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                mel_scale="slaney",
            ).to(self.device)
            self.transforms.append(transform)

    def _load_audio(self, filepath):
        """
        Loads audio from disk, resamples, mixes to mono, and pads/crops to fixed length.
        """
        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        # Handle potential missing files safely (though metadata validation ensures they exist)
        if not os.path.exists(full_path):
            return torch.zeros(1, Config.N_SAMPLES)

        try:
            waveform, sr = torchaudio.load(full_path)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, Config.N_SAMPLES)

        # Resample if necessary
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

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
        eps = 1e-9  # Epsilon for log stability

        for transform in self.transforms:
            # Compute Mel Spectrogram: Shape (1, n_mels, time)
            mel_spec = transform(waveform)

            # Apply Log scale
            log_mel = torch.log(mel_spec + eps)
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
