import os
import glob
import random
import numpy as np
import torch
import torchaudio
from library.config import Config


def load_noise_bank(load_cached_data: bool = True):
    """
    Loads background noise files from the input directory.
    Implements caching using .npz files to avoid repeated I/O.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "noise_bank.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Convert back to list of tensors
            noise_bank = [torch.from_numpy(data[f]) for f in sorted(data.files)]
            return noise_bank
        except Exception:
            # If load fails, fall back to processing from scratch
            pass

    # 2. Load from source
    noise_dir = os.path.join(Config.TRAIN_AUDIO_DIR, "_background_noise_")
    noise_bank = []

    if os.path.exists(noise_dir):
        files = glob.glob(os.path.join(noise_dir, "*.wav"))
        for f in files:
            try:
                # Load waveform
                waveform, sr = torchaudio.load(f)

                # Resample if needed
                if sr != Config.SR:
                    resampler = torchaudio.transforms.Resample(sr, Config.SR)
                    waveform = resampler(waveform)

                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                noise_bank.append(waveform)
            except Exception:
                continue

    # 3. Save to cache
    if noise_bank:
        # Convert to numpy for storage
        # We use a dict with keys arr_0, arr_1, etc.
        noise_dict = {f"arr_{i}": w.numpy() for i, w in enumerate(noise_bank)}
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.savez(cache_path, **noise_dict)

    return noise_bank


class DualChannelProcessor:
    """
    Handles waveform processing, augmentation, and dual-channel spectrogram generation.
    """

    def __init__(self, load_cached_data: bool = True):
        # Load noise bank for augmentation and silence synthesis
        self.noise_bank = load_noise_bank(load_cached_data)

        # Channel 1: High Frequency Resolution (Formants)
        # Window size ~64ms (1024 samples)
        self.mel_freq = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT_FREQ,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
            center=True,
            normalized=False,
        )

        # Channel 2: High Temporal Resolution (Transients)
        # Window size ~25ms (400 samples)
        self.mel_time = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT_TIME,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
            center=True,
            normalized=False,
        )

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    def _get_noise_crop(self, length: int) -> torch.Tensor:
        """
        Extracts a random crop of specified length from the noise bank.
        """
        if not self.noise_bank:
            return torch.zeros(1, length)

        noise_wav = random.choice(self.noise_bank)
        c, t = noise_wav.shape

        if t <= length:
            # Repeat to fill if noise clip is shorter than target
            repeats = (length // t) + 1
            noise_wav = noise_wav.repeat(1, repeats)
            t = noise_wav.shape[1]

        start = random.randint(0, t - length)
        return noise_wav[:, start : start + length]

    def augment_waveform(
        self, waveform: torch.Tensor, snr_min: float, snr_max: float
    ) -> torch.Tensor:
        """
        Mixes background noise into the waveform with a random SNR.
        """
        # Calculate signal power
        sig_pow = waveform.pow(2).mean()
        if sig_pow == 0:
            return waveform

        # Get noise crop
        noise = self._get_noise_crop(waveform.shape[1])
        noise_pow = noise.pow(2).mean()
        if noise_pow == 0:
            return waveform

        # Determine target SNR and coefficient
        snr = random.uniform(snr_min, snr_max)
        target_ratio = 10 ** (snr / 10)

        # coeff = sqrt(P_sig / (P_noise * ratio))
        coeff = torch.sqrt(sig_pow / (noise_pow * target_ratio))

        return waveform + coeff * noise

    def __call__(
        self, waveform: torch.Tensor, mode: str = "train", label: str = None
    ) -> torch.Tensor:
        """
        Processes a raw waveform into a Dual-Channel Spectrogram.

        Args:
            waveform (torch.Tensor): Input waveform (1, Time).
            mode (str): 'train' or 'val'.
            label (str): Label of the sample (used for silence synthesis).

        Returns:
            torch.Tensor: (2, 128, 101) tensor.
        """
        target_len = Config.AUDIO_LEN_SAMPLES

        # 1. Dynamic Silence Synthesis
        # If label is silence, ignore input and generate fresh noise
        if label == Config.SILENCE_LABEL and mode == "train":
            waveform = self._get_noise_crop(target_len)
        else:
            # 2. Length Standardization (Pad/Crop)
            c, t = waveform.shape
            if t > target_len:
                if mode == "train":
                    # Random crop for training
                    start = random.randint(0, t - target_len)
                    waveform = waveform[:, start : start + target_len]
                else:
                    # Center crop for validation/test
                    start = (t - target_len) // 2
                    waveform = waveform[:, start : start + target_len]
            elif t < target_len:
                # Pad with zeros
                padding = target_len - t
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # 3. Augmentation (Noise Injection)
        # Apply to non-silence samples during training
        if mode == "train" and label != Config.SILENCE_LABEL:
            if random.random() < Config.NOISE_PROB:
                waveform = self.augment_waveform(
                    waveform, Config.NOISE_SNR_MIN, Config.NOISE_SNR_MAX
                )

        # 4. Dual Spectrogram Generation
        # Channel 1: Frequency Focus
        spec_freq = self.mel_freq(waveform)  # (1, 128, T)

        # Channel 2: Time Focus
        spec_time = self.mel_time(waveform)  # (1, 128, T)

        # Convert to dB
        spec_freq = self.amp_to_db(spec_freq)
        spec_time = self.amp_to_db(spec_time)

        # Remove channel dim (1, F, T) -> (F, T)
        spec_freq = spec_freq.squeeze(0)
        spec_time = spec_time.squeeze(0)

        # Stack into (2, F, T)
        dual_spec = torch.stack([spec_freq, spec_time], dim=0)

        # 5. Instance Normalization
        # Normalize per sample: (x - mean) / std
        mean = dual_spec.mean(dim=(1, 2), keepdim=True)
        std = dual_spec.std(dim=(1, 2), keepdim=True) + 1e-5
        dual_spec = (dual_spec - mean) / std

        return dual_spec
