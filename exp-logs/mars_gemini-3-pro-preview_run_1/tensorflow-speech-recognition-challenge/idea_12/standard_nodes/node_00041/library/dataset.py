import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import FineGrainedLabelEncoder, get_metadata


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpeechCommandDataset(Dataset):
    def __init__(self, dataframe, label_encoder, mode="train", noise_files=None):
        """
        Args:
            dataframe (pd.DataFrame): Metadata containing 'filepath' and 'label'.
            label_encoder (FineGrainedLabelEncoder): Encoder for labels.
            mode (str): 'train', 'val', or 'test'.
            noise_files (list): List of paths to background noise wav files.
        """
        self.df = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.mode = mode
        self.noise_files = noise_files if noise_files else []

        # Audio settings
        self.sr = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.num_samples = Config.NUM_SAMPLES

        # Spectrogram transforms
        # MelSpectrogram returns power spectrogram by default (power=2.0)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Preload noise files for efficiency
        self.noise_waves = []
        if (self.mode == "train" or self.mode == "val") and self.noise_files:
            for nf in self.noise_files:
                try:
                    # Load noise file
                    w, s = torchaudio.load(nf)
                    # Resample if necessary
                    if s != self.sr:
                        resampler = torchaudio.transforms.Resample(s, self.sr)
                        w = resampler(w)
                    # Mix to mono
                    if w.shape[0] > 1:
                        w = torch.mean(w, dim=0, keepdim=True)
                    self.noise_waves.append(w)
                except Exception as e:
                    pass

    def __len__(self):
        return len(self.df)

    def _get_audio(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"]

        # 1. Handle Silence (Generation from Background Noise)
        if label == "silence":
            if not self.noise_waves:
                return torch.zeros(1, self.num_samples)

            # For validation, we want determinism based on index
            if self.mode != "train":
                rng = random.Random(idx)
                noise_idx = rng.randint(0, len(self.noise_waves) - 1)
                noise_wave = self.noise_waves[noise_idx]

                if noise_wave.shape[1] <= self.num_samples:
                    pad_amt = self.num_samples - noise_wave.shape[1]
                    waveform = torch.nn.functional.pad(noise_wave, (0, pad_amt))
                else:
                    max_start = noise_wave.shape[1] - self.num_samples
                    start = rng.randint(0, max_start)
                    waveform = noise_wave[:, start : start + self.num_samples]
            else:
                # Training: Random crop
                noise_idx = random.randint(0, len(self.noise_waves) - 1)
                noise_wave = self.noise_waves[noise_idx]

                if noise_wave.shape[1] <= self.num_samples:
                    pad_amt = self.num_samples - noise_wave.shape[1]
                    waveform = torch.nn.functional.pad(noise_wave, (0, pad_amt))
                else:
                    max_start = noise_wave.shape[1] - self.num_samples
                    start = random.randint(0, max_start)
                    waveform = noise_wave[:, start : start + self.num_samples]

            # Random volume for silence
            vol = random.uniform(0.0, 1.0) if self.mode == "train" else 1.0
            return waveform * vol

        # 2. Handle Regular Audio
        full_path = os.path.join(Config.INPUT_ROOT, filepath)
        if not os.path.exists(full_path):
            return torch.zeros(1, self.num_samples)

        waveform, sr = torchaudio.load(full_path)

        # Resample
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Crop
        if waveform.shape[1] < self.num_samples:
            pad_amt = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        elif waveform.shape[1] > self.num_samples:
            if self.mode == "train":
                max_start = waveform.shape[1] - self.num_samples
                start = random.randint(0, max_start)
                waveform = waveform[:, start : start + self.num_samples]
            else:
                # Center crop for val/test
                start = (waveform.shape[1] - self.num_samples) // 2
                waveform = waveform[:, start : start + self.num_samples]

        return waveform

    def _inject_noise(self, waveform):
        if not self.noise_waves or random.random() > Config.NOISE_PROB:
            return waveform

        noise_idx = random.randint(0, len(self.noise_waves) - 1)
        noise_wave = self.noise_waves[noise_idx]

        # Get a chunk of noise
        if noise_wave.shape[1] <= self.num_samples:
            pad_amt = self.num_samples - noise_wave.shape[1]
            noise_chunk = torch.nn.functional.pad(noise_wave, (0, pad_amt))
        else:
            start = random.randint(0, noise_wave.shape[1] - self.num_samples)
            noise_chunk = noise_wave[:, start : start + self.num_samples]

        # Calculate SNR
        signal_energy = torch.mean(waveform**2)
        noise_energy = torch.mean(noise_chunk**2)

        if noise_energy < 1e-9:
            return waveform

        snr = random.uniform(Config.NOISE_SNR_MIN, Config.NOISE_SNR_MAX)
        target_noise_energy = signal_energy / (10 ** (snr / 10))
        scale = torch.sqrt(target_noise_energy / noise_energy)

        return waveform + noise_chunk * scale

    def __getitem__(self, idx):
        # 1. Get Waveform
        waveform = self._get_audio(idx)

        # 2. Augmentation (Noise Injection)
        if self.mode == "train":
            waveform = self._inject_noise(waveform)

        # 3. Spectrogram (Power Spectrogram)
        mel_spec = self.mel_transform(waveform)

        # 4. Energy Calculation (Frame-wise RMS from Spectrogram)
        # Input mel_spec is (1, n_mels, time). We average over n_mels.
        # This gives a proxy for frame energy.
        energy = torch.mean(mel_spec, dim=1, keepdim=True)  # (1, 1, T)

        # 5. Log Mel
        log_mel = self.db_transform(mel_spec)

        # 6. Label
        label_str = self.df.iloc[idx]["label"]
        if self.mode == "test":
            label_id = -1
        else:
            try:
                label_id = self.label_encoder.transform([label_str])[0]
            except:
                label_id = 0  # Fallback

        # Return: Spec (1, F, T), Energy (T), Label (int)
        return log_mel, energy.squeeze(), label_id


def get_dataloaders(load_cached_data=True):
    set_seed(Config.SEED)

    # 1. Load Metadata
    df_train_raw = get_metadata("train")
    df_val = get_metadata("val")
    df_test = get_metadata("test")

    # 2. Initialize Encoder
    label_encoder = FineGrainedLabelEncoder()
    label_encoder.fit(df_train_raw)

    # 3. Prepare Balanced Training Data
    cache_path = os.path.join(Config.WORK_DIR, "train_balanced.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached balanced training data from {cache_path}")
        df_train_balanced = pd.read_parquet(cache_path)
    else:
        print("Balancing training data...")
        records = []

        target_labels = Config.TARGET_LABELS

        # Separate data
        df_targets = df_train_raw[df_train_raw["label"].isin(target_labels)]
        df_aux = df_train_raw[
            (~df_train_raw["label"].isin(target_labels))
            & (df_train_raw["label"] != "silence")
        ]

        # A. Upsample Targets (to ~2000 each)
        TARGET_COUNT = 2000
        for label in target_labels:
            df_class = df_targets[df_targets["label"] == label]
            if len(df_class) == 0:
                continue

            if len(df_class) < TARGET_COUNT:
                df_class = df_class.sample(
                    n=TARGET_COUNT, replace=True, random_state=Config.SEED
                )
            else:
                df_class = df_class.sample(
                    n=TARGET_COUNT, replace=False, random_state=Config.SEED
                )
            records.append(df_class)

        # B. Downsample Aux (to ~9000 total)
        AUX_TOTAL = 9000
        if len(df_aux) > AUX_TOTAL:
            df_aux_sampled = df_aux.sample(
                n=AUX_TOTAL, replace=False, random_state=Config.SEED
            )
            records.append(df_aux_sampled)
        else:
            records.append(df_aux)

        # C. Create Silence Samples (2000 samples)
        SILENCE_COUNT = 2000
        silence_records = pd.DataFrame(
            {
                "filepath": ["SILENCE_GENERATED"] * SILENCE_COUNT,
                "label": ["silence"] * SILENCE_COUNT,
                "subject_id": ["generated"] * SILENCE_COUNT,
                "is_noise": [True] * SILENCE_COUNT,
            }
        )
        records.append(silence_records)

        df_train_balanced = pd.concat(records, ignore_index=True)

        # Save cache
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        df_train_balanced.to_parquet(cache_path)
        print(f"Saved balanced training data to {cache_path}")

    # 4. Identify Noise Files for Augmentation
    noise_df = df_train_raw[df_train_raw["label"] == "silence"]
    noise_files = [
        os.path.join(Config.INPUT_ROOT, f) for f in noise_df["filepath"].tolist()
    ]
    noise_files = [f for f in noise_files if os.path.exists(f)]

    # 5. Create Datasets
    train_dataset = SpeechCommandDataset(
        df_train_balanced, label_encoder, mode="train", noise_files=noise_files
    )

    val_dataset = SpeechCommandDataset(
        df_val, label_encoder, mode="val", noise_files=noise_files
    )

    test_dataset = SpeechCommandDataset(
        df_test, label_encoder, mode="test", noise_files=None
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_encoder
