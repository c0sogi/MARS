import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal
from torch.utils.data import Dataset
from library import config
from library import utils


class SpeechCommandsDataset(Dataset):
    def __init__(self, metadata_df, phase="train", transform=None):
        """
        PyTorch Dataset for Speech Commands.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'filepath', 'label', etc.
            phase (str): 'train', 'val', or 'test'. Controls balancing and specific loading logic.
            transform (callable, optional): Optional transform to apply to the spectrogram.
        """
        self.phase = phase
        self.transform = transform
        self.featurizer = utils.get_featurizer()

        # Cache for full silence waveforms to avoid repeated I/O during training
        self.silence_cache = {}

        # Process and balance DataFrame based on phase
        self.df = self._process_dataframe(metadata_df)

        # Preload silence files only if we are in training phase for random cropping
        if self.phase == "train":
            self._preload_silence_files()

    def _process_dataframe(self, df):
        """
        Balances the training set by undersampling unknowns and oversampling silence.
        Leaves val/test datasets as is to maintain evaluation integrity.
        """
        if self.phase != "train":
            return df.reset_index(drop=True)

        # Separate classes
        df_silence = df[df["label"] == "silence"]
        df_unknown = df[df["label"] == "unknown"]
        df_targets = df[~df["label"].isin(["silence", "unknown"])]

        # Target count: approx 2000 samples per class to match target commands
        target_count = 2000

        # 1. Undersample Unknowns (down to ~2000)
        if len(df_unknown) > target_count:
            df_unknown = df_unknown.sample(n=target_count, random_state=config.SEED)

        # 2. Oversample Silence (up to ~2000)
        # Replicate rows so __getitem__ is called enough times to generate diverse crops
        if len(df_silence) > 0:
            factor = int(np.ceil(target_count / len(df_silence)))
            df_silence = pd.concat([df_silence] * factor, ignore_index=True)
            df_silence = df_silence.iloc[:target_count]

        # Combine and Shuffle
        df_balanced = pd.concat([df_targets, df_unknown, df_silence], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=config.SEED).reset_index(
            drop=True
        )

        print(f"[{self.phase.upper()}] Dataset Balanced:")
        print(f"  Total: {len(df_balanced)}")
        print(f"  Targets: {len(df_targets)}")
        print(f"  Unknown: {len(df_unknown)}")
        print(f"  Silence: {len(df_silence)}")

        return df_balanced

    def _preload_silence_files(self):
        """
        Loads full background noise files into memory for fast random cropping.
        Uses soundfile and scipy to avoid torchaudio dependency.
        """
        silence_paths = self.df[self.df["label"] == "silence"]["filepath"].unique()
        for filepath in silence_paths:
            full_path = os.path.join(config.INPUT_DIR, filepath)
            if not os.path.exists(full_path):
                continue

            try:
                # Load full file
                audio_data, sr = sf.read(full_path)
                audio_data = audio_data.astype(np.float32)

                # Mix to mono
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Resample if necessary
                if sr != config.SAMPLE_RATE:
                    num_samples = int(len(audio_data) * config.SAMPLE_RATE / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                    audio_data = audio_data.astype(np.float32)

                # Convert to Tensor (1, Time)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)

                self.silence_cache[filepath] = waveform
            except Exception as e:
                print(f"Warning: Failed to preload silence file {filepath}: {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        # 1. Audio Loading
        # If Training and Silence: Random Crop from cached full file
        if self.phase == "train" and label_str == "silence":
            if filepath in self.silence_cache:
                full_wav = self.silence_cache[filepath]
                # Random crop
                if full_wav.shape[1] > config.AUDIO_LEN:
                    max_start = full_wav.shape[1] - config.AUDIO_LEN
                    start = torch.randint(0, max_start + 1, (1,)).item()
                    waveform = full_wav[:, start : start + config.AUDIO_LEN]
                else:
                    # Pad if short (unlikely for background noise files)
                    pad_amt = config.AUDIO_LEN - full_wav.shape[1]
                    waveform = torch.nn.functional.pad(full_wav, (0, pad_amt))
            else:
                # Fallback to utils if not cached (should not happen if preloaded correctly)
                waveform = utils.load_and_pad_audio(filepath)
        else:
            # Deterministic load (uses disk cache from utils)
            # Used for Targets, Unknowns, Val/Test sets, and fallback
            waveform = utils.load_and_pad_audio(filepath)

        # 2. Featurization
        # waveform shape: (1, 16000) -> spec shape: (1, n_mels, time)
        spec = self.featurizer(waveform)

        # Convert to Log-Mel Spectrogram
        spec = (spec + 1e-6).log()

        # Apply optional transforms
        if self.transform:
            spec = self.transform(spec)

        # 3. Label Processing
        # Map label string to index. 'unknown' is default for unseen labels.
        label_idx = config.LABEL_TO_IDX.get(label_str, config.LABEL_TO_IDX["unknown"])

        return spec, label_idx
