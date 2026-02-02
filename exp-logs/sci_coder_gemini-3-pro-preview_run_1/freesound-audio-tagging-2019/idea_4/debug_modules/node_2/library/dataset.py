import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config


class AudioDataset(Dataset):
    """
    PyTorch Dataset for Audio Tagging.
    Handles loading, preprocessing, spectrogram generation, and augmentation.
    """

    def __init__(self, split="train", debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, use a small subset of data.
        """
        self.split = split
        self.sr = Config.SR
        self.n_mels = Config.N_MELS
        self.duration = Config.TRAIN_DURATION if split == "train" else None

        # 1. Load Metadata
        if split == "train":
            csv_path = Config.TRAIN_CSV
        elif split == "val":
            csv_path = Config.VAL_CSV
        elif split == "test":
            csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Handle Debug Mode
        if debug or Config.DEBUG:
            df = df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)

        self.file_paths = df["filepath"].tolist()

        # 2. Extract Labels
        # Identify class columns (exclude metadata)
        meta_cols = {"fname", "labels", "filepath"}
        self.label_cols = [c for c in df.columns if c not in meta_cols]

        # Verify class count matches config
        if len(self.label_cols) != Config.NUM_CLASSES:
            # In case of mismatch, we trust the Config and slice or warn
            # But here we expect strict alignment with metadata generation
            pass

        # Convert labels to float32 tensor
        # For test set, these are placeholders (zeros)
        self.labels = df[self.label_cols].values.astype(np.float32)

        # 3. Initialize Transforms
        # Audio preprocessing
        # EDA showed source is 44100Hz, target is 32000Hz
        self.resampler = T.Resample(orig_freq=44100, new_freq=self.sr)

        # Spectrogram extraction
        self.mel_spec = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=self.n_mels,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )

        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        # Augmentations (Train only)
        if self.split == "train":
            self.time_masking = T.TimeMasking(time_mask_param=Config.SPEC_AUG_TIME_MASK)
            self.freq_masking = T.FrequencyMasking(
                freq_mask_param=Config.SPEC_AUG_FREQ_MASK
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load Audio
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        try:
            # Load waveform: (channels, time)
            waveform, sr = torchaudio.load(full_path)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return silent tensor as fallback
            target_len = int(self.sr * (self.duration if self.duration else 5.0))
            waveform = torch.zeros(1, target_len)
            sr = self.sr

        # 2. Resample
        if sr != self.sr:
            if sr == 44100:
                waveform = self.resampler(waveform)
            else:
                # Fallback for unexpected sample rates
                resampler = T.Resample(sr, self.sr)
                waveform = resampler(waveform)

        # 3. Mix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Crop or Pad (Training Only)
        # Validation/Test uses full length
        if self.split == "train" and self.duration is not None:
            target_len = int(self.sr * self.duration)
            current_len = waveform.shape[1]

            if current_len > target_len:
                # Random Crop
                start = random.randint(0, current_len - target_len)
                waveform = waveform[:, start : start + target_len]
            elif current_len < target_len:
                # Pad by tiling (repeating) the audio to fill the context
                # This is often better than zero-padding for short sound events
                num_repeats = int(np.ceil(target_len / current_len))
                waveform = waveform.repeat(1, num_repeats)
                waveform = waveform[:, :target_len]

        # 5. Compute Log-Mel Spectrogram
        # Shape: (1, n_mels, time)
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        # 6. Apply SpecAugment (Training Only)
        if self.split == "train":
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 7. Input Repetition
        # Convert (1, F, T) -> (3, F, T) to match ImageNet pretrained backbone
        if Config.USE_INPUT_REPETITION:
            spec = spec.repeat(3, 1, 1)

        # 8. Prepare Target
        label = self.labels[idx]
        target = torch.tensor(label, dtype=torch.float32)

        return spec, target
