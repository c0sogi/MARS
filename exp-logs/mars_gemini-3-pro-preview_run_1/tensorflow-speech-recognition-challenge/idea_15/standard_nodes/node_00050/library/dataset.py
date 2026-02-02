import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_ROOT,
    WORKING_DIR,
    AUDIO_PARAMS,
    TRAINING_PARAMS,
    LABEL2ID,
    TARGET_LABELS_SET,
    get_fine_grained_label_from_path,
    set_seed,
)

# Global cache for background noise waveforms to avoid repeated disk I/O
# This is safe because the total size of background noise files is small (<20MB)
NOISE_CACHE = {}


class SpeechCommandsDataset(Dataset):
    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'filepath' and 'label'.
            phase (str): 'train', 'val', or 'test'. Controls augmentation and label handling.
        """
        self.df = df
        self.phase = phase
        self.target_length = int(AUDIO_PARAMS["sample_rate"] * AUDIO_PARAMS["duration"])

        # Load background noise files into memory
        self._load_noise_files()

        # Audio Transforms
        # 1. Mel Spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_PARAMS["sample_rate"],
            n_fft=AUDIO_PARAMS["n_fft"],
            hop_length=AUDIO_PARAMS["hop_length"],
            n_mels=AUDIO_PARAMS["n_mels"],
            f_min=AUDIO_PARAMS["f_min"],
            f_max=AUDIO_PARAMS["f_max"],
        )
        # 2. Amplitude to DB
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            top_db=AUDIO_PARAMS["top_db"]
        )

        # Augmentation Transforms (SpecAugment)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

    def _load_noise_files(self):
        """Loads all .wav files from the _background_noise_ directory into memory."""
        noise_dir = os.path.join(INPUT_ROOT, "train", "audio", "_background_noise_")
        if not os.path.exists(noise_dir):
            return

        if not NOISE_CACHE:
            for f in os.listdir(noise_dir):
                if f.endswith(".wav"):
                    path = os.path.join(noise_dir, f)
                    # Load waveform
                    waveform, sr = torchaudio.load(path)

                    # Convert to Mono if necessary
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    # Resample if necessary
                    if sr != AUDIO_PARAMS["sample_rate"]:
                        resampler = torchaudio.transforms.Resample(
                            sr, AUDIO_PARAMS["sample_rate"]
                        )
                        waveform = resampler(waveform)

                    NOISE_CACHE[f] = waveform

    def _get_silence_sample(self):
        """Generates a silence sample by randomly cropping a background noise file."""
        if not NOISE_CACHE:
            return torch.zeros(1, self.target_length)

        # Pick a random noise file
        noise_key = np.random.choice(list(NOISE_CACHE.keys()))
        noise_wav = NOISE_CACHE[noise_key]

        # Random crop
        if noise_wav.shape[1] > self.target_length:
            start = np.random.randint(0, noise_wav.shape[1] - self.target_length)
            return noise_wav[:, start : start + self.target_length]
        else:
            # Pad if too short
            padding = self.target_length - noise_wav.shape[1]
            return torch.nn.functional.pad(noise_wav, (0, padding))

    def _load_audio(self, filepath, label):
        """Loads audio, handling silence generation and padding/cropping."""
        # Dynamic Silence Synthesis
        if label == "silence":
            return self._get_silence_sample()

        full_path = os.path.join(INPUT_ROOT, filepath)

        # Load file
        try:
            waveform, sr = torchaudio.load(full_path)
        except Exception:
            # Fallback for corrupted files (should be rare given metadata check)
            return torch.zeros(1, self.target_length)

        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != AUDIO_PARAMS["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, AUDIO_PARAMS["sample_rate"])
            waveform = resampler(waveform)

        # Adjust Length (Pad or Crop)
        current_len = waveform.shape[1]
        if current_len < self.target_length:
            padding = self.target_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            # Center crop for consistency
            start = (current_len - self.target_length) // 2
            waveform = waveform[:, start : start + self.target_length]

        return waveform

    def _inject_noise(self, waveform):
        """Injects background noise into the waveform with random SNR."""
        if (
            not NOISE_CACHE
            or np.random.rand() > TRAINING_PARAMS["noise_injection_prob"]
        ):
            return waveform

        # Pick random noise
        noise_key = np.random.choice(list(NOISE_CACHE.keys()))
        noise_wav = NOISE_CACHE[noise_key]

        # Get random chunk
        if noise_wav.shape[1] <= self.target_length:
            noise_chunk = torch.nn.functional.pad(
                noise_wav, (0, self.target_length - noise_wav.shape[1])
            )
        else:
            start = np.random.randint(0, noise_wav.shape[1] - self.target_length)
            noise_chunk = noise_wav[:, start : start + self.target_length]

        # Calculate SNR scaling
        snr_db = np.random.uniform(
            TRAINING_PARAMS["noise_snr_min"], TRAINING_PARAMS["noise_snr_max"]
        )

        speech_rms = waveform.pow(2).mean().sqrt()
        noise_rms = noise_chunk.pow(2).mean().sqrt()

        if noise_rms > 0:
            target_noise_rms = speech_rms / (10 ** (snr_db / 20))
            scale = target_noise_rms / noise_rms
            waveform = waveform + (noise_chunk * scale)

        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # Determine Label
        # For Test set, we don't have labels, return dummy
        if self.phase == "test":
            fine_label = "unknown"
            label_idx = 0
        else:
            # Extract fine-grained label from path (e.g., 'bed', 'silence')
            fine_label = get_fine_grained_label_from_path(filepath)
            label_idx = LABEL2ID.get(fine_label, LABEL2ID["silence"])  # Fallback safe

        # Load Audio
        waveform = self._load_audio(filepath, fine_label)

        # Waveform Augmentation (Train only)
        if self.phase == "train":
            waveform = self._inject_noise(waveform)

        # Generate Spectrogram
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)

        # Spectrogram Augmentation (Train only)
        if (
            self.phase == "train"
            and np.random.rand() < TRAINING_PARAMS["spec_augment_prob"]
        ):
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        return spec, label_idx


def get_dataloaders(batch_size, num_workers, load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Implements Variance-Aware Balancing for the training set.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    train_cache_path = os.path.join(WORKING_DIR, "train_balanced.parquet")

    # ---------------------------------------------------------
    # 1. Prepare Training Data (Balanced)
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(train_cache_path):
        df_train = pd.read_parquet(train_cache_path)
    else:
        df_raw = pd.read_csv(TRAIN_METADATA_PATH)

        # Add fine-grained label column for grouping
        df_raw["fine_label"] = df_raw["filepath"].apply(
            get_fine_grained_label_from_path
        )

        balanced_records = []
        target_count = TRAINING_PARAMS["target_sample_count"]

        # Group by label to apply specific balancing strategies
        groups = df_raw.groupby("fine_label")

        for label, group in groups:
            if label in TARGET_LABELS_SET:
                # Strategy: Upsample Target Commands to target_count
                # We use replace=True to duplicate samples if count < target_count
                # We also limit to target_count if count > target_count (though rare for this dataset)
                resampled = group.sample(
                    n=target_count, replace=True, random_state=TRAINING_PARAMS["seed"]
                )
                balanced_records.append(resampled)

            elif label == "silence":
                # Strategy: Upsample Silence
                # Silence has very few files (background noise). We replicate the rows many times.
                # The Dataset class will randomly crop these files, creating diverse samples.
                resampled = group.sample(
                    n=target_count, replace=True, random_state=TRAINING_PARAMS["seed"]
                )
                balanced_records.append(resampled)

            else:
                # Strategy: Keep Auxiliary Classes Natural
                # We do not upsample 'bed', 'bird', etc.
                # Preserving their natural distribution maintains high variance for the 'unknown' class.
                balanced_records.append(group)

        df_train = pd.concat(balanced_records).reset_index(drop=True)

        # Cache the result
        df_train.to_parquet(train_cache_path)

    # ---------------------------------------------------------
    # 2. Prepare Validation and Test Data
    # ---------------------------------------------------------
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # ---------------------------------------------------------
    # 3. Create Datasets and Loaders
    # ---------------------------------------------------------
    train_dataset = SpeechCommandsDataset(df_train, phase="train")
    val_dataset = SpeechCommandsDataset(df_val, phase="val")
    test_dataset = SpeechCommandsDataset(df_test, phase="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
