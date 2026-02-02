import os
import random
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class AudioDataset(Dataset):
    def __init__(self, split, load_cached_data=True, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            debug (bool): If True, restricts dataset to a small subset.
        """
        self.split = split
        self.sr = Config.SR
        self.duration = Config.DURATION
        self.target_length = self.sr * self.duration
        self.debug = debug

        # Define cache path for metadata
        self.cache_dir = os.path.join(Config.WORKING_DIR, "dataset_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"meta_{split}.parquet")

        # 1. Load Metadata
        if load_cached_data and os.path.exists(cache_path):
            self.df = pd.read_parquet(cache_path)
        else:
            # Determine source file
            if split == "train":
                src_path = Config.TRAIN_METADATA
            elif split == "val":
                src_path = Config.VAL_METADATA
            elif split == "test":
                src_path = Config.TEST_METADATA
            else:
                raise ValueError(f"Invalid split: {split}")

            self.df = pd.read_csv(src_path)

            # Pre-process labels for train/val
            if split != "test":
                # Load class names from sample submission to ensure correct order
                sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
                self.classes = [c for c in sub_df.columns if c != "fname"]
                class_to_idx = {c: i for i, c in enumerate(self.classes)}

                # Function to encode labels
                def encode_labels(label_str):
                    # Create binary vector
                    vec = np.zeros(len(self.classes), dtype=np.int8)
                    if pd.isna(label_str) or label_str == "":
                        return vec
                    for lbl in label_str.split(","):
                        if lbl in class_to_idx:
                            vec[class_to_idx[lbl]] = 1
                    return vec

                # Apply encoding
                # We store vectors as lists in the dataframe to be parquet-compatible
                self.df["target_vec"] = self.df["labels"].apply(
                    lambda x: encode_labels(x).tolist()
                )

            # Save to cache
            self.df.to_parquet(cache_path)

        # Debug subset
        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)

        # 2. Prepare Targets (for fast access in __getitem__)
        if split != "test":
            # Convert list of lists back to tensor
            self.targets = torch.tensor(
                np.vstack(self.df["target_vec"].tolist()), dtype=torch.float32
            )

        # 3. Audio Transforms
        # Common source SR is 44100 based on analysis, target is 32000
        self.resampler_44k = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=self.sr
        )

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        # compute_deltas removed Cite solution_lesson_node_00015

        # 4. Augmentations (Train only)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        # 1. Load Audio
        try:
            wav, org_sr = torchaudio.load(filepath)
        except Exception:
            # Fallback for read errors (should be rare given metadata filtering)
            wav = torch.zeros(1, self.target_length)
            org_sr = self.sr

        # 2. Resample
        if org_sr != self.sr:
            if org_sr == 44100:
                wav = self.resampler_44k(wav)
            else:
                resampler = torchaudio.transforms.Resample(org_sr, self.sr)
                wav = resampler(wav)

        # 3. Mix to Mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # 4. Fix Duration (Pad or Truncate)
        num_samples = wav.shape[1]
        if num_samples < self.target_length:
            # Pad with zeros at the end
            pad_amt = self.target_length - num_samples
            wav = torch.nn.functional.pad(wav, (0, pad_amt))
        elif num_samples > self.target_length:
            # Truncate
            if self.split == "train":
                # Random crop for training
                start = random.randint(0, num_samples - self.target_length)
                wav = wav[:, start : start + self.target_length]
            else:
                # Center crop or simple truncation for val/test?
                # Simple truncation (keep start) is more deterministic for validation
                wav = wav[:, : self.target_length]

        # 5. Feature Extraction
        # Channel 1: Log-Mel Spectrogram
        mel = self.mel_spec(wav)
        log_mel = self.amp_to_db(mel)  # Shape: (1, n_mels, time)

        # Apply SpecAugment
        if self.split == "train":
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)

        # Use single channel Cite solution_lesson_node_00015
        image = log_mel

        # 6. Normalization (Instance-wise Standardization)
        # Normalize per sample to handle varying recording volumes
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True)
        image = (image - mean) / (std + 1e-6)

        # Return
        if self.split == "test":
            return image, fname
        else:
            target = self.targets[idx]
            return image, target
