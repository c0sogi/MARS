import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SAMPLE_RATE,
    DURATION,
    AUDIO_LEN,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    F_MIN,
    F_MAX,
    LABEL2ID,
    NOISE_SNR_MIN,
    NOISE_SNR_MAX,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    TARGET_LABELS,
    SILENCE_LABEL,
    UNKNOWN_LABEL,
    get_source_label,
)

# Caching path
CACHE_PATH = os.path.join(WORKING_DIR, "train_balanced_v2.parquet")


class SpeechDataset(Dataset):
    def __init__(self, df, mode="train", noise_files=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            noise_files (list of torch.Tensor): Preloaded noise waveforms for injection.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.noise_files = noise_files if noise_files else []

        # Audio Feature Extraction
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentation
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(INPUT_DIR, row["filepath"])

        # 1. Load Audio
        waveform = self._load_audio(filepath)

        # 2. Determine Label
        if self.mode == "test":
            label_id = -1  # Dummy for test
            label_str = UNKNOWN_LABEL
        else:
            # Re-derive fine-grained label from filepath
            label_str = get_source_label(row["filepath"])
            label_id = LABEL2ID.get(label_str, LABEL2ID[UNKNOWN_LABEL])

        # 3. Noise Injection (Train only, exclude silence class)
        # We don't add noise to silence because silence IS noise.
        if self.mode == "train" and self.noise_files and label_str != SILENCE_LABEL:
            if torch.rand(1).item() < 0.8:  # 80% probability
                waveform = self._inject_noise(waveform)

        # 4. Generate Spectrogram
        # waveform shape: (1, Time) -> spec shape: (1, Freq, Time)
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)

        # 5. SpecAugment (Train only)
        if self.mode == "train":
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        # Ensure shape (1, Freq, Time)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec, torch.tensor(label_id, dtype=torch.long)

    def _load_audio(self, filepath):
        try:
            wav, sr = torchaudio.load(filepath)
        except Exception:
            # Fallback for corrupt files
            wav = torch.zeros(1, AUDIO_LEN)
            sr = SAMPLE_RATE

        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        # Convert to Mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Pad or Crop to fixed length
        curr_len = wav.shape[1]
        if curr_len < AUDIO_LEN:
            pad_len = AUDIO_LEN - curr_len
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        elif curr_len > AUDIO_LEN:
            # Random crop for training (essential for silence synthesis), Center for val/test
            if self.mode == "train":
                offset = torch.randint(0, curr_len - AUDIO_LEN + 1, (1,)).item()
            else:
                offset = (curr_len - AUDIO_LEN) // 2
            wav = wav[:, offset : offset + AUDIO_LEN]

        return wav

    def _inject_noise(self, waveform):
        # Select random noise file
        noise = self.noise_files[torch.randint(0, len(self.noise_files), (1,)).item()]

        # Crop noise to match waveform length
        noise_len = noise.shape[1]
        if noise_len > AUDIO_LEN:
            offset = torch.randint(0, noise_len - AUDIO_LEN + 1, (1,)).item()
            noise_crop = noise[:, offset : offset + AUDIO_LEN]
        else:
            pad_len = AUDIO_LEN - noise_len
            noise_crop = torch.nn.functional.pad(noise, (0, pad_len))

        # Calculate SNR scaling
        signal_power = waveform.pow(2).mean()
        noise_power = noise_crop.pow(2).mean()

        if noise_power == 0 or signal_power == 0:
            return waveform

        snr_db = torch.empty(1).uniform_(NOISE_SNR_MIN, NOISE_SNR_MAX).item()
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_power / noise_power)

        return waveform + scale * noise_crop


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Implements Variance-Aware Balancing and Caching.
    """
    # 0. Load Background Noise Files for Injection
    noise_dir = os.path.join(INPUT_DIR, "train", "audio", "_background_noise_")
    noise_files = []
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith(".wav"):
                try:
                    w, s = torchaudio.load(os.path.join(noise_dir, f))
                    if s != SAMPLE_RATE:
                        w = torchaudio.transforms.Resample(s, SAMPLE_RATE)(w)
                    # Ensure mono
                    if w.shape[0] > 1:
                        w = torch.mean(w, dim=0, keepdim=True)
                    noise_files.append(w)
                except:
                    continue

    # 1. Prepare Training Data (with Balancing and Caching)
    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached balanced training data from {CACHE_PATH}...")
        df_train = pd.read_parquet(CACHE_PATH)
    else:
        print("Constructing balanced training dataset...")
        df_raw = pd.read_csv(TRAIN_METADATA_PATH)

        # Extract fine-grained labels for balancing
        df_raw["fine_label"] = df_raw["filepath"].apply(get_source_label)

        # Split into groups
        # Targets: yes, no, up, down...
        # Silence: _background_noise_ files
        # Aux: bed, bird, etc.

        df_targets = df_raw[df_raw["fine_label"].isin(TARGET_LABELS)]
        df_silence = df_raw[df_raw["fine_label"] == SILENCE_LABEL]
        df_aux = df_raw[
            (~df_raw["fine_label"].isin(TARGET_LABELS))
            & (df_raw["fine_label"] != SILENCE_LABEL)
        ]

        balanced_dfs = []

        # A. Upsample Targets to ~2000
        TARGET_COUNT = 2000
        for label in TARGET_LABELS:
            sub = df_targets[df_targets["fine_label"] == label]
            if len(sub) == 0:
                continue

            if len(sub) < TARGET_COUNT:
                sub = sub.sample(n=TARGET_COUNT, replace=True, random_state=42)
            else:
                sub = sub.sample(n=TARGET_COUNT, replace=False, random_state=42)
            balanced_dfs.append(sub)

        # B. Upsample Silence to ~2000
        # Since silence files are long and we random crop, duplicating rows
        # effectively creates new samples.
        if len(df_silence) > 0:
            if len(df_silence) < TARGET_COUNT:
                sub_silence = df_silence.sample(
                    n=TARGET_COUNT, replace=True, random_state=42
                )
            else:
                sub_silence = df_silence.sample(
                    n=TARGET_COUNT, replace=False, random_state=42
                )
            balanced_dfs.append(sub_silence)

        # C. Downsample Aux to 5x Target Count (Variance-Aware Balancing)
        # Cite solution_lesson_node_00026: 1:5 ratio between specific targets and Unknown class
        AUX_COUNT = TARGET_COUNT * 5
        if len(df_aux) > AUX_COUNT:
            sub_aux = df_aux.sample(n=AUX_COUNT, replace=False, random_state=42)
            balanced_dfs.append(sub_aux)
        else:
            balanced_dfs.append(df_aux)

        # Combine and Shuffle
        df_train = pd.concat(balanced_dfs, ignore_index=True)
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

        # Cache
        os.makedirs(WORKING_DIR, exist_ok=True)
        df_train.to_parquet(CACHE_PATH)
        print(f"Balanced training data saved to {CACHE_PATH} ({len(df_train)} samples)")

    # 2. Validation Data
    df_val = pd.read_csv(VAL_METADATA_PATH)

    # 3. Test Data
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # 4. Create Datasets
    train_ds = SpeechDataset(df_train, mode="train", noise_files=noise_files)
    val_ds = SpeechDataset(df_val, mode="val")
    test_ds = SpeechDataset(df_test, mode="test")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
