import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class SpeechCommandDataset(Dataset):
    def __init__(self, metadata_df, input_root, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            input_root (str): Root directory for input data.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df
        self.input_root = input_root
        self.mode = mode

        # Define MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
        )

        # SpecAugment transforms
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.metadata)

    def _load_audio(self, file_path, is_background):
        full_path = os.path.join(self.input_root, file_path)

        # Load audio using soundfile
        try:
            audio, sr = sf.read(full_path)
        except Exception:
            # Return silence if file is corrupted
            return torch.zeros(Config.NUM_SAMPLES)

        # Handle multi-channel audio (convert to mono)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        audio_tensor = torch.from_numpy(audio).float()
        audio_len = audio_tensor.shape[0]
        target_len = Config.NUM_SAMPLES

        # Logic for Silence (Background Noise) in Training
        if self.mode == "train" and is_background:
            if audio_len > target_len:
                # Random crop for background noise
                start_idx = torch.randint(0, audio_len - target_len, (1,)).item()
                audio_tensor = audio_tensor[start_idx : start_idx + target_len]
            else:
                padding = target_len - audio_len
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        else:
            # Logic for Commands or Test/Val (Center Crop or Pad)
            if audio_len > target_len:
                start_idx = (audio_len - target_len) // 2
                audio_tensor = audio_tensor[start_idx : start_idx + target_len]
            elif audio_len < target_len:
                # Right padding
                padding = target_len - audio_len
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        return audio_tensor

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row["file_path"]
        label_str = row["label"]

        # Check if background noise
        is_background = False
        if "is_background" in row:
            is_background = row["is_background"]

        # 1. Load and process audio
        waveform = self._load_audio(file_path, is_background)

        # 2. Feature Extraction (MelSpectrogram)
        spec = self.mel_transform(waveform)

        # 3. Log Transform
        log_spec = torch.log(spec + 1e-9)

        # 4. SpecAugment (only in train mode)
        if self.mode == "train":
            log_spec = self.freq_masking(log_spec)
            log_spec = self.time_masking(log_spec)

        # 5. Label
        if label_str in Config.LABEL2IDX:
            label = Config.LABEL2IDX[label_str]
        else:
            label = Config.LABEL2IDX.get("unknown", 0)

        # Return: input (1, freq, time), label
        return log_spec.unsqueeze(0), label


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train and validation sets with weighted sampling.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Datasets
    train_dataset = SpeechCommandDataset(df_train, Config.INPUT_ROOT, mode="train")
    val_dataset = SpeechCommandDataset(df_val, Config.INPUT_ROOT, mode="val")

    # 3. Weighted Sampler for Training to handle class imbalance
    class_counts = df_train["label"].value_counts().to_dict()
    train_labels = df_train["label"].values

    sample_weights = []
    for label in train_labels:
        if label in class_counts:
            sample_weights.append(1.0 / class_counts[label])
        else:
            sample_weights.append(0.0)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for test set.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = SpeechCommandDataset(df_test, Config.INPUT_ROOT, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
