import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Handles loading raw audio, preprocessing to Log-Mel Spectrograms,
    caching processed data to disk, and applying augmentations.
    """

    def __init__(self, split, load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load preprocessed data from cache.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Determine metadata file based on split
        if self.split == "train":
            self.csv_path = Config.TRAIN_CSV
        elif self.split == "val":
            self.csv_path = Config.VAL_CSV
        elif self.split == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        # Cache file path (e.g., ./working/idea_2/train.npz)
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{self.split}.npz")

        # --- Define Audio Preprocessing Transforms ---
        # 1. Mel Spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )

        # 2. Amplitude to dB (Log Scale)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # --- Define Augmentation Transforms (Train only) ---
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK
        )

        # Execute loading logic
        self._load_data()

    def _load_data(self):
        """
        Loads data from cache if available, otherwise processes raw audio and caches it.
        """
        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"[{self.split}] Loading cached data from {self.cache_path}...")
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self.specs = data["specs"]
                self.labels = data["labels"]
                self.clip_names = data["clip_names"]
                print(
                    f"[{self.split}] Successfully loaded {len(self.specs)} samples from cache."
                )
                return
            except Exception as e:
                print(
                    f"[{self.split}] Error loading cache: {e}. Falling back to processing from scratch."
                )

        # 2. Process from scratch
        print(f"[{self.split}] Processing raw audio files from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        specs_list = []
        labels_list = []
        clip_names_list = []

        # Iterate over metadata
        for _, row in df.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            clip_name = row["clip_name"]

            # Get label (default to 0.0 for test set where label column is missing)
            label = float(row["label"]) if "label" in row else 0.0

            try:
                # Load Audio
                # torchaudio.load returns (waveform, sample_rate)
                waveform, sr = torchaudio.load(full_path)

                # Resample if necessary
                if sr != Config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=Config.SAMPLE_RATE
                    )
                    waveform = resampler(waveform)

                # Convert to Mono if necessary (average channels)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Pad or Truncate to fixed duration
                target_len = Config.N_SAMPLES
                current_len = waveform.shape[1]

                if current_len < target_len:
                    # Pad with zeros at the end
                    pad_amount = target_len - current_len
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                elif current_len > target_len:
                    # Truncate
                    waveform = waveform[:, :target_len]

                # Compute Log-Mel Spectrogram
                spec = self.mel_transform(waveform)
                spec = self.db_transform(spec)

                # Append to lists (store as numpy to save memory/disk)
                specs_list.append(spec.numpy())
                labels_list.append(label)
                clip_names_list.append(clip_name)

            except Exception as e:
                print(f"[{self.split}] Error processing {full_path}: {e}")
                # Insert dummy data to maintain alignment (crucial for test set submission)
                # Shape: [1, n_mels, time_frames]
                # time_frames approx N_SAMPLES // HOP_LENGTH + 1
                dummy_time_dim = (
                    spec.shape[2]
                    if "spec" in locals()
                    else int(Config.N_SAMPLES / Config.HOP_LENGTH) + 1
                )
                dummy_spec = np.zeros(
                    (1, Config.N_MELS, dummy_time_dim), dtype=np.float32
                )
                specs_list.append(dummy_spec)
                labels_list.append(label)
                clip_names_list.append(clip_name)

        # Convert lists to numpy arrays
        # Stacking specs creates a single array [N, 1, F, T]
        self.specs = np.stack(specs_list).astype(np.float32)
        self.labels = np.array(labels_list, dtype=np.float32)
        self.clip_names = np.array(clip_names_list)

        # Save to cache
        print(f"[{self.split}] Saving processed data to {self.cache_path}...")
        np.savez(
            self.cache_path,
            specs=self.specs,
            labels=self.labels,
            clip_names=self.clip_names,
        )
        print(f"[{self.split}] Processing complete. Saved {len(self.specs)} samples.")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        """
        Returns:
            spec (torch.Tensor): Preprocessed spectrogram [Channels, Freq, Time]
            label (torch.Tensor): Binary label float32
        """
        # Retrieve data
        spec_np = self.specs[idx]
        label_val = self.labels[idx]

        # Convert to Tensor
        spec = torch.from_numpy(spec_np)
        label = torch.tensor(label_val, dtype=torch.float32)

        # Apply Augmentations (Train only)
        if self.split == "train":
            # SpecAugment: Masking blocks of frequency or time
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # Normalization (Instance-level Standard Scaling)
        # This makes the model robust to global volume differences between clips
        mean = spec.mean()
        std = spec.std()
        # Avoid division by zero
        spec = (spec - mean) / (std + 1e-6)

        return spec, label

    def get_clip_name(self, idx):
        """
        Helper to retrieve the clip name for a specific index.
        Useful for generating submission files.
        """
        return self.clip_names[idx]
