import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from library.config import Config


class SpeechCommandsDataset(Dataset):
    """
    Dataset class for Speech Command Recognition implementing Idea 12.
    Features:
    - On-the-fly waveform loading.
    - Dynamic Waveform Noise Injection (Train only).
    - High-Fidelity Log-Mel Spectrogram generation (1024 FFT, 128 Mels).
    - Instance Normalization.
    - SpecAugment (Train only).
    """

    def __init__(self, subset="train", load_cached_data=True):
        """
        Args:
            subset (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Flag for caching logic (kept for signature compatibility,
                                     though this pipeline is dynamic).
        """
        self.subset = subset
        self.sr = Config.SAMPLE_RATE
        self.target_length = Config.NUM_SAMPLES  # 16000 samples (1s)

        # Load Metadata
        if subset == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.training = True
        elif subset == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.training = False
        elif subset == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            self.training = False
        else:
            raise ValueError(f"Unknown subset: {subset}")

        # Debug mode: subset data
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)

        # Pre-load background noise for injection (only for training)
        self.bg_noises = []
        if self.training:
            self._load_background_noises()

        # Audio Transforms
        # MelSpectrogram: 1024 FFT, 128 Mels, 25ms window, 10ms hop
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            power=2.0,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # SpecAugment Transforms
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )

    def _load_background_noises(self):
        """
        Loads all background noise files into memory for fast mixing.
        """
        if not os.path.exists(Config.BACKGROUND_NOISE_DIR):
            return

        for filename in os.listdir(Config.BACKGROUND_NOISE_DIR):
            if filename.endswith(".wav"):
                path = os.path.join(Config.BACKGROUND_NOISE_DIR, filename)
                try:
                    wav, sr = sf.read(path)
                    if sr != self.sr:
                        # Simple resampling if needed, though dataset analysis says all are 16k
                        # For robustness, we skip or would need resampling.
                        # Given analysis, we assume 16k.
                        pass
                    self.bg_noises.append(wav)
                except Exception:
                    pass

    def _load_audio(self, file_path, is_silence_label=False):
        """
        Loads audio, pads/crops to target length.
        """
        full_path = os.path.join(Config.INPUT_ROOT, file_path)

        try:
            wav, sr = sf.read(full_path)
        except Exception:
            # Fallback for corrupted files: return silence
            return np.zeros(self.target_length, dtype=np.float32)

        # If it's a 'silence' label (background noise file), it is likely long.
        # We need to crop a random 1s segment for training, or a specific segment for val.
        # For consistency, we treat 'silence' label files as sources to be cropped.

        curr_len = len(wav)

        if curr_len > self.target_length:
            if self.training or is_silence_label:
                # Random crop
                start = np.random.randint(0, curr_len - self.target_length)
            else:
                # Center crop for validation/test of normal clips (if any are > 1s)
                start = (curr_len - self.target_length) // 2

            wav = wav[start : start + self.target_length]

        elif curr_len < self.target_length:
            # Pad with zeros
            pad_len = self.target_length - curr_len
            # Symmetric padding or end padding? End padding is standard for commands.
            wav = np.pad(wav, (0, pad_len), mode="constant")

        return wav.astype(np.float32)

    def _inject_noise(self, wav):
        """
        Mixes the signal with a random background noise segment.
        """
        if not self.bg_noises or np.random.random() > Config.NOISE_INJECTION_PROB:
            return wav

        # Select random noise file
        noise = random.choice(self.bg_noises)
        noise_len = len(noise)

        if noise_len < self.target_length:
            return wav  # Skip if noise is too short

        # Random crop of noise
        start = np.random.randint(0, noise_len - self.target_length)
        noise_segment = noise[start : start + self.target_length]

        # Calculate RMS
        sig_rms = np.sqrt(np.mean(wav**2))
        noise_rms = np.sqrt(np.mean(noise_segment**2))

        if noise_rms < 1e-9:
            return wav

        # Select random SNR
        snr_db = np.random.uniform(Config.NOISE_MIN_SNR_DB, Config.NOISE_MAX_SNR_DB)
        snr_factor = 10 ** (snr_db / 20)

        # Scale noise
        target_noise_rms = sig_rms / snr_factor
        scaled_noise = noise_segment * (target_noise_rms / noise_rms)

        # Mix
        return wav + scaled_noise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        label_str = row["label"]

        # Determine if this sample represents the 'silence' class
        is_silence = label_str == "silence"

        # 1. Load Waveform
        wav = self._load_audio(file_path, is_silence_label=is_silence)

        # 2. Waveform Augmentation (Noise Injection)
        # We generally don't inject noise into 'silence' class as it is already noise.
        if self.training and not is_silence:
            wav = self._inject_noise(wav)

        # 3. Convert to Tensor
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, Time)

        # 4. Spectrogram Generation
        spec = self.melspec_transform(wav_tensor)
        spec = self.db_transform(spec)  # (1, n_mels, time)

        # 5. Instance Normalization
        # Standardize per sample to mean=0, std=1
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 6. SpecAugment (Training Only)
        if self.training:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 7. Label Encoding
        label_idx = Config.LABEL2IDX.get(label_str, Config.LABEL2IDX["unknown"])

        return spec, label_idx
