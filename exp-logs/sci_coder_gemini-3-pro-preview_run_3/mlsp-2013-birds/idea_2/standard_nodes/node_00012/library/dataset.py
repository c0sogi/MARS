import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import RandomAffine, ColorJitter
from library.config import Config


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Reads raw WAV files and generates Log-Mel Spectrograms on-the-fly.
    """

    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'labels'.
            phase (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.target_length = Config.SR * Config.DURATION

        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)

        # SpecAugment Transforms
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPECAUG_FREQ_MASK_PARAM
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPECAUG_TIME_MASK_PARAM
        )

        # Image Transforms (Cite solution_lesson_node_00007, solution_lesson_node_00009)
        self.affine_transform = RandomAffine(degrees=0, translate=Config.AUG_TRANSLATE)
        self.jitter_transform = ColorJitter(
            brightness=Config.AUG_BRIGHTNESS, contrast=Config.AUG_CONTRAST
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Audio
        # Construct full path: ./input + essential_data/src_wavs/...
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        try:
            # Load audio (soundfile returns numpy array)
            wav, sr = sf.read(full_path)

            # Ensure correct sampling rate (though dataset is 16k)
            if sr != Config.SR:
                pass

        except Exception as e:
            # Fallback for corrupt files (return silence)
            wav = np.zeros(self.target_length, dtype=np.float32)

        # 2. Fix Length (Pad or Crop)
        if len(wav) < self.target_length:
            padding = self.target_length - len(wav)
            wav = np.pad(wav, (0, padding), mode="constant")
        elif len(wav) > self.target_length:
            if self.phase == "train":
                start = random.randint(0, len(wav) - self.target_length)
            else:
                start = (len(wav) - self.target_length) // 2
            wav = wav[start : start + self.target_length]

        # Convert to Tensor
        wav_tensor = torch.from_numpy(wav).float()

        # 3. Generate Log-Mel Spectrogram
        # Shape: (n_mels, time)
        spec = self.mel_transform(wav_tensor)
        spec = self.db_transform(spec)

        # 4. Normalization (Min-Max per instance to [0, 1])
        # Normalize BEFORE augmentations so ColorJitter works on [0, 1] range
        spec_min = spec.min()
        spec_max = spec.max()
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)

        # Add Channel Dimension: (1, n_mels, time)
        spec = spec.unsqueeze(0)

        # 5. Spectrogram Augmentations (Train Only)
        if self.phase == "train":
            # SpecAugment
            if random.random() < 0.5:
                spec = self.freq_masking(spec)
            if random.random() < 0.5:
                spec = self.time_masking(spec)

            # Horizontal Translation (Time Shift) - Cite solution_lesson_node_00007
            spec = self.affine_transform(spec)

            # Photometric Augmentation (Brightness/Contrast) - Cite solution_lesson_node_00009
            spec = self.jitter_transform(spec)

        # 7. Parse Labels
        label_vec = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        label_str = str(row["labels"])
        if label_str != "?" and label_str.lower() != "nan" and label_str.strip():
            try:
                indices = [int(x) for x in label_str.split()]
                for i in indices:
                    if 0 <= i < Config.NUM_CLASSES:
                        label_vec[i] = 1.0
            except ValueError:
                pass  # Handle empty or malformed labels gracefully

        return spec, label_vec


def load_dataset_df(split="train", load_cached_data=False):
    """
    Loads the dataframe for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Unused here as loading CSV is fast,
                                 but kept for signature compatibility.

    Returns:
        pd.DataFrame: The requested dataframe.
    """
    if split == "train":
        path = Config.TRAIN_CSV
    elif split == "val":
        path = Config.VAL_CSV
    elif split == "test":
        path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df
