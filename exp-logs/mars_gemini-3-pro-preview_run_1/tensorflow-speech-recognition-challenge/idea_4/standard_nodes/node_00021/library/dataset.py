import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from library.config import Config
from library.utils import set_seed


class SpeechCommandDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Speech Command Recognition.
    Handles data balancing, dynamic silence generation, spectrogram extraction, and SpecAugment.
    """

    def __init__(self, mode="train", config=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            config (Config): Configuration object containing paths and hyperparameters.
            load_cached_data (bool): Whether to load balanced metadata from cache.
        """
        self.mode = mode
        self.config = config if config is not None else Config()
        self.load_cached_data = load_cached_data

        # Ensure reproducibility
        set_seed(self.config.seed)

        # Pre-initialize transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Augmentation transforms
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.config.freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=self.config.time_mask_param
        )

        # Load Data
        self.df = self._prepare_data()

        # Load background noise for silence generation (Train/Val only)
        self.background_noises = []
        if self.mode in ["train", "val"]:
            self._load_background_noise()

    def _prepare_data(self):
        """
        Loads metadata and performs balancing for the training set.
        Implements caching for the balanced training dataframe.
        """
        # Determine paths
        if self.mode == "train":
            metadata_path = self.config.train_metadata_path
            cache_path = os.path.join(self.config.working_dir, "train_balanced.parquet")
        elif self.mode == "val":
            metadata_path = self.config.val_metadata_path
            cache_path = None
        else:
            metadata_path = self.config.test_metadata_path
            cache_path = None

        # Load raw metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # For validation and test, just load and return (maybe subset for debug)
        if self.mode != "train":
            df = pd.read_csv(metadata_path)
            if self.config.subset_size:
                df = df.sample(
                    n=min(len(df), self.config.subset_size),
                    random_state=self.config.seed,
                ).reset_index(drop=True)
            return df

        # --- Training Data Balancing Logic with Caching ---

        # 1. Try to load cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # Apply subset if debugging
                if self.config.subset_size:
                    df = df.sample(
                        n=min(len(df), self.config.subset_size),
                        random_state=self.config.seed,
                    ).reset_index(drop=True)
                return df
            except Exception:
                pass  # Fallback to re-computing

        # 2. Compute from scratch
        df_raw = pd.read_csv(metadata_path)

        # Filter out noise files from main list (handled separately)
        # In metadata generation, background noise has is_noise=True or label='silence' and specific paths
        # We rely on label='silence' to identify them.
        # However, for the balanced dataset, we want to generate silence dynamically.
        # So we remove existing 'silence' entries from the dataframe used for file loading.
        df_clean = df_raw[df_raw["label"] != "silence"].copy()

        # Define target count per class
        # We aim for ~2000 samples per class to balance 'unknown' and targets
        target_count = 2000

        balanced_dfs = []

        # Process Target Labels
        for label in self.config.target_labels:
            df_label = df_clean[df_clean["label"] == label]
            if len(df_label) == 0:
                continue
            # Upsample with replacement
            df_resampled = df_label.sample(
                n=target_count, replace=True, random_state=self.config.seed
            )
            balanced_dfs.append(df_resampled)

        # Process Unknown Label
        df_unknown = df_clean[df_clean["label"] == "unknown"]
        if not df_unknown.empty:
            # Downsample without replacement if possible, else with replacement
            replace = len(df_unknown) < target_count
            df_resampled = df_unknown.sample(
                n=target_count, replace=replace, random_state=self.config.seed
            )
            balanced_dfs.append(df_resampled)

        # Process Silence (Add Placeholders)
        # We add rows with a special filepath that __getitem__ recognizes
        silence_data = {
            "filepath": ["SILENCE_PLACEHOLDER"] * target_count,
            "label": ["silence"] * target_count,
            "subject_id": ["synthetic"] * target_count,
        }
        balanced_dfs.append(pd.DataFrame(silence_data))

        # Combine and Shuffle
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_balanced.to_parquet(cache_path, index=False)

        # Apply subset if debugging
        if self.config.subset_size:
            df_balanced = df_balanced.sample(
                n=min(len(df_balanced), self.config.subset_size),
                random_state=self.config.seed,
            ).reset_index(drop=True)

        return df_balanced

    def _load_background_noise(self):
        """
        Loads background noise files into memory for dynamic silence generation.
        """
        noise_dir = os.path.join(self.config.train_dir, "_background_noise_")
        if not os.path.exists(noise_dir):
            return

        noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]

        for f in noise_files:
            path = os.path.join(noise_dir, f)
            try:
                waveform, sr = torchaudio.load(path)
                # Resample if necessary
                if sr != self.config.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        sr, self.config.sample_rate
                    )
                    waveform = resampler(waveform)

                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                self.background_noises.append(waveform)
            except Exception:
                continue

    def _get_silence_sample(self):
        """
        Generates a 1-second silence sample by cropping background noise.
        If no noise files loaded, returns zeros.
        """
        if not self.background_noises:
            return torch.zeros(1, self.config.n_samples)

        # Select random noise file
        noise_idx = random.randint(0, len(self.background_noises) - 1)
        noise_wav = self.background_noises[noise_idx]

        noise_len = noise_wav.shape[1]
        if noise_len <= self.config.n_samples:
            # If noise file is short, pad it
            padding = self.config.n_samples - noise_len
            return torch.nn.functional.pad(noise_wav, (0, padding))

        # Random crop
        start_idx = random.randint(0, noise_len - self.config.n_samples)
        return noise_wav[:, start_idx : start_idx + self.config.n_samples]

    def _process_waveform(self, waveform):
        """
        Ensures waveform is mono and exactly 1 second long.
        """
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Fix length
        length = waveform.shape[1]
        if length < self.config.n_samples:
            padding = self.config.n_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.config.n_samples:
            waveform = waveform[:, : self.config.n_samples]

        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = row["label"]
        filepath = row["filepath"]

        # 1. Load Waveform
        if label_str == "silence" and filepath == "SILENCE_PLACEHOLDER":
            waveform = self._get_silence_sample()
        else:
            full_path = os.path.join(self.config.input_root, filepath)
            try:
                waveform, sr = torchaudio.load(full_path)
                if sr != self.config.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        sr, self.config.sample_rate
                    )
                    waveform = resampler(waveform)
                waveform = self._process_waveform(waveform)
            except Exception:
                # Fallback for corrupted files: return silence
                waveform = torch.zeros(1, self.config.n_samples)

        # 1.5. Random Gain Augmentation (Train only)
        # Cite Failure Analysis: Correlation between low intensity and error.
        if self.mode == "train":
            gain = random.uniform(0.5, 1.5)
            waveform = waveform * gain

        # 2. Generate Spectrogram
        # waveform shape: [1, n_samples]
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)

        # 3. Augmentation (Train only)
        if self.mode == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

            # Optional: Add random noise to non-silence samples?
            # (Already using Mixup in training loop, so maybe skip here to avoid over-regularization)

        # 4. Prepare Label
        # For test set, label might be dummy, but we return it anyway
        # If label is 'unknown' in test set, we map it to 'unknown' index
        # The Config.label2id handles 'unknown' and 'silence'
        label_id = self.config.label2id.get(label_str, self.config.label2id["unknown"])

        return spec, label_id
