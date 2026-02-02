import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


class SpeechCommandDataset(Dataset):
    def __init__(self, df, mode="train", config=Config):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.config = config
        self.audio_len = config.AUDIO_LEN

        # Audio Transforms
        # MelSpectrogram expects waveform of shape (..., time)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentations (Train only)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

        # Cache background noise for silence synthesis if needed
        self.noise_cache = {}
        # We cache noise files if we are in training or validation and there are silence labels
        if self.mode in ["train", "val"]:
            self._cache_noise_files()

    def _cache_noise_files(self):
        """
        Pre-loads background noise files into memory to avoid high I/O during training.
        """
        noise_rows = self.df[self.df["label"] == "silence"]
        if noise_rows.empty:
            return

        unique_noise_paths = noise_rows["filepath"].unique()

        for rel_path in unique_noise_paths:
            full_path = os.path.join(self.config.INPUT_ROOT, rel_path)
            if os.path.exists(full_path):
                try:
                    # Load waveform
                    waveform, sr = torchaudio.load(full_path)

                    # Convert to Mono if necessary
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # Resample if necessary
                    if sr != self.config.SAMPLE_RATE:
                        resampler = torchaudio.transforms.Resample(
                            sr, self.config.SAMPLE_RATE
                        )
                        waveform = resampler(waveform)

                    self.noise_cache[rel_path] = waveform
                except Exception as e:
                    # In case of error, we just won't cache it and handle it in get_audio
                    pass

    def _get_audio(self, row):
        """
        Loads audio, handles silence generation, pads/trims to fixed length.
        """
        label = row["label"]
        filepath = row["filepath"]

        # 1. Handle Silence (Background Noise)
        if label == "silence" and filepath in self.noise_cache:
            noise_wave = self.noise_cache[filepath]
            noise_len = noise_wave.shape[1]

            if noise_len > self.audio_len:
                if self.mode == "train":
                    # Random crop for training
                    start = torch.randint(0, noise_len - self.audio_len, (1,)).item()
                else:
                    # Center crop for validation (deterministic)
                    start = (noise_len - self.audio_len) // 2

                waveform = noise_wave[:, start : start + self.audio_len]
            else:
                # Pad if noise file is shorter than 1s (unlikely)
                padding = self.audio_len - noise_len
                waveform = torch.nn.functional.pad(noise_wave, (0, padding))

        # 2. Handle Standard Files
        else:
            full_path = os.path.join(self.config.INPUT_ROOT, filepath)

            if not os.path.exists(full_path):
                # Return silent tensor if file missing
                return torch.zeros(1, self.audio_len)

            try:
                waveform, sr = torchaudio.load(full_path)
            except:
                return torch.zeros(1, self.audio_len)

            # Convert to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample
            if sr != self.config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Pad or Trim
            sig_len = waveform.shape[1]
            if sig_len < self.audio_len:
                # Pad with zeros at the end
                padding = self.audio_len - sig_len
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif sig_len > self.audio_len:
                # Trim
                if self.mode == "train":
                    # Center crop is safer for speech commands to avoid cutting off start/end
                    start = (sig_len - self.audio_len) // 2
                    waveform = waveform[:, start : start + self.audio_len]
                else:
                    # Center crop
                    start = (sig_len - self.audio_len) // 2
                    waveform = waveform[:, start : start + self.audio_len]

        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Audio
        waveform = self._get_audio(row)

        # 2. Generate Spectrogram
        # waveform shape: (1, 16000) -> spec shape: (1, 128, 101)
        spec = self.mel_spectrogram(waveform)

        # 3. Log Scale (dB)
        spec = self.amplitude_to_db(spec)

        # 4. Augmentation (Train Only)
        if self.mode == "train":
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 5. Normalization (Instance Level)
        # Standardize to mean 0, std 1 per sample
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 6. Label Encoding
        label_str = row["label"]
        # Map label to ID. If not in map (shouldn't happen for train/val), map to unknown.
        # For test, label is 'unknown' placeholder.
        label_id = self.config.LABEL2ID.get(label_str, self.config.LABEL2ID["unknown"])

        return spec, torch.tensor(label_id, dtype=torch.long)


def get_dataset(mode, config=Config, load_cached_data=True):
    """
    Factory function to create the dataset.
    Handles data balancing and caching for the training set.
    """
    if mode == "train":
        cache_path = os.path.join(config.WORKING_DIR, "train_balanced.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df_balanced = pd.read_parquet(cache_path)
                return SpeechCommandDataset(df_balanced, mode="train", config=config)
            except Exception:
                pass  # Fallback to compute

        # 2. Compute Balanced Dataset
        df = pd.read_csv(config.TRAIN_METADATA_PATH)

        # Separate classes
        df_silence = df[df["label"] == "silence"]
        df_unknown = df[df["label"] == "unknown"]
        df_targets = df[(df["label"] != "silence") & (df["label"] != "unknown")]

        # Determine target count (Median of target classes to avoid extreme upsampling)
        target_counts = df_targets["label"].value_counts()
        target_n = int(target_counts.median())

        balanced_dfs = []

        # A. Targets: Upsample/Downsample to target_n
        for label in config.TARGET_LABELS:
            df_cls = df_targets[df_targets["label"] == label]
            if len(df_cls) == 0:
                continue

            if len(df_cls) < target_n:
                # Upsample
                df_cls = df_cls.sample(
                    n=target_n, replace=True, random_state=config.SEED
                )
            else:
                # Downsample
                df_cls = df_cls.sample(
                    n=target_n, replace=False, random_state=config.SEED
                )
            balanced_dfs.append(df_cls)

        # B. Unknown: Downsample to target_n
        if len(df_unknown) > 0:
            count = min(len(df_unknown), target_n)
            df_unknown = df_unknown.sample(
                n=count, replace=False, random_state=config.SEED
            )
            balanced_dfs.append(df_unknown)

        # C. Silence: Upsample to target_n (re-using the few noise files)
        if len(df_silence) > 0:
            df_silence = df_silence.sample(
                n=target_n, replace=True, random_state=config.SEED
            )
            balanced_dfs.append(df_silence)

        # Combine and Shuffle
        df_balanced = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=config.SEED)
            .reset_index(drop=True)
        )

        # 3. Save Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_balanced.to_parquet(cache_path, index=False)

        return SpeechCommandDataset(df_balanced, mode="train", config=config)

    elif mode == "val":
        df = pd.read_csv(config.VAL_METADATA_PATH)
        return SpeechCommandDataset(df, mode="val", config=config)

    elif mode == "test":
        df = pd.read_csv(config.TEST_METADATA_PATH)
        return SpeechCommandDataset(df, mode="test", config=config)

    else:
        raise ValueError(f"Unknown mode: {mode}")
