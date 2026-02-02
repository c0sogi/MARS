import os
import random
import torch
import torchaudio
import numpy as np
from library.config import Config


class AudioProcessor:
    """
    Handles audio loading, preprocessing, augmentation, and feature extraction.
    """

    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.num_samples = Config.NUM_SAMPLES
        self.device = torch.device("cpu")  # Datasets usually load on CPU

        # Spectrogram Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment Transforms
        # Parameters chosen for 1s clips with 128 mels
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

        # Load background noises into memory
        self.noises = self._load_background_noises()

    def _load_background_noises(self):
        """
        Loads all .wav files from the _background_noise_ directory.
        Returns a list of waveforms.
        """
        noise_dir = os.path.join(Config.TRAIN_AUDIO_DIR, "_background_noise_")
        noises = []
        if not os.path.exists(noise_dir):
            return noises

        for filename in os.listdir(noise_dir):
            if filename.endswith(".wav"):
                path = os.path.join(noise_dir, filename)
                try:
                    waveform, sr = torchaudio.load(path)
                    # Resample if necessary (though usually they are 16k)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)

                    # Ensure mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    noises.append(waveform)
                except Exception as e:
                    print(f"Warning: Failed to load noise file {filename}: {e}")
        return noises

    def load_waveform(self, filepath):
        """
        Loads an audio file, converts to mono, and pads/truncates to fixed length.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            torch.Tensor: Waveform of shape (1, NUM_SAMPLES)
        """
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception:
            # Return silence if file is corrupt
            return torch.zeros(1, self.num_samples)

        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Truncate
        waveform_len = waveform.shape[1]
        if waveform_len < self.num_samples:
            # Pad with zeros
            padding = self.num_samples - waveform_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform_len > self.num_samples:
            # Truncate (center crop is safer for commands, but simple slice is standard)
            # For consistency with standard implementations, we take the beginning
            waveform = waveform[:, : self.num_samples]

        return waveform

    def add_background_noise(self, waveform, snr_min=10, snr_max=30):
        """
        Injects real background noise into the waveform.

        Args:
            waveform (torch.Tensor): Input waveform (1, L).
            snr_min (int): Minimum SNR in dB.
            snr_max (int): Maximum SNR in dB.

        Returns:
            torch.Tensor: Noisy waveform.
        """
        if not self.noises:
            return waveform

        noise = random.choice(self.noises)
        noise_len = noise.shape[1]
        wave_len = waveform.shape[1]

        # Extract a random segment of noise
        if noise_len > wave_len:
            start = random.randint(0, noise_len - wave_len)
            noise_segment = noise[:, start : start + wave_len]
        else:
            # Tile noise if it's shorter than waveform
            repeats = 1 + (wave_len // noise_len)
            noise_segment = noise.repeat(1, repeats)[:, :wave_len]

        # Calculate power
        signal_power = waveform.pow(2).mean()
        noise_power = noise_segment.pow(2).mean()

        if noise_power == 0 or signal_power == 0:
            return waveform

        # Determine target SNR
        snr = random.uniform(snr_min, snr_max)

        # Calculate scale factor
        # SNR = 10 * log10(P_signal / (scale * P_noise)^2) -> P_target_noise = P_signal / 10^(SNR/10)
        target_noise_power = signal_power / (10 ** (snr / 10))
        scale = torch.sqrt(target_noise_power / noise_power)

        return waveform + scale * noise_segment

    def get_spectrogram(self, waveform):
        """
        Converts waveform to Log-Mel Spectrogram.

        Args:
            waveform (torch.Tensor): (1, L)

        Returns:
            torch.Tensor: (1, n_mels, time_steps)
        """
        # Compute Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to DB (Log scale)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec

    def apply_spec_augment(self, spectrogram):
        """
        Applies Time and Frequency masking.

        Args:
            spectrogram (torch.Tensor): (1, n_mels, time_steps)

        Returns:
            torch.Tensor: Masked spectrogram.
        """
        # SpecAugment expects (..., freq, time)
        # Torchaudio transforms handle the channel dim correctly if it's (C, F, T)
        masked = self.freq_masking(spectrogram)
        masked = self.time_masking(masked)
        return masked

    def process_audio(self, filepath, is_training=False, should_augment=False):
        """
        Full pipeline: Load -> (Noise) -> Spectrogram -> (SpecAugment).

        Args:
            filepath (str): Path to audio file.
            is_training (bool): If True, enables augmentation logic.
            should_augment (bool): Explicit flag to enable/disable augmentation
                                   (useful if we want training mode but no aug for some reason).

        Returns:
            torch.Tensor: Final tensor ready for the model (1, F, T).
        """
        # 1. Load Waveform
        waveform = self.load_waveform(filepath)

        # 2. Waveform Augmentation (Noise Injection)
        if is_training and should_augment:
            # Apply with 80% probability
            if random.random() < 0.8:
                waveform = self.add_background_noise(waveform)

        # 3. Generate Spectrogram
        spectrogram = self.get_spectrogram(waveform)

        # 4. Spectrogram Augmentation (SpecAugment)
        if is_training and should_augment:
            spectrogram = self.apply_spec_augment(spectrogram)

        return spectrogram
