import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to a batch of data.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup beta distribution parameter.
        device (str or torch.device): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input data.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class AudioDataset(Dataset):
    def __init__(self, csv_path, mode="train"):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and data splitting.
        """
        self.mode = mode
        self.csv_path = csv_path

        # Load Metadata
        self.df = pd.read_csv(csv_path)

        # Debugging: Subset data if configured
        if Config.debug:
            # Use a fixed subset for reproducibility in debug mode
            self.df = self.df.iloc[: Config.debug_sample_size]

        # Load Class Mappings from sample_submission.csv to ensure correct order
        self.classes = self._get_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Audio Transforms
        # 1. Mel Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels,
            f_min=Config.fmin,
            f_max=Config.fmax,
        )

        # 2. Amplitude to DB (Log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 3. Augmentations (SpecAugment) - Only used in training
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.spec_augment_time_mask_param
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.spec_augment_freq_mask_param
        )

    def _get_classes(self):
        """Reads the sample submission file to get the list of classes in correct order."""
        if not os.path.exists(Config.sample_submission_path):
            raise FileNotFoundError(
                f"Sample submission not found at {Config.sample_submission_path}"
            )

        df = pd.read_csv(Config.sample_submission_path)
        # Columns are: fname, Label1, Label2, ...
        # We exclude 'fname'
        classes = [c for c in df.columns if c != "fname"]
        return classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Audio
        # Metadata filepath is relative to input directory
        file_path = os.path.join(Config.input_root, row["filepath"])

        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Fallback for read errors: return silent tensor
            # This ensures training doesn't crash on a single bad file
            waveform = torch.zeros(1, Config.target_length)
            sr = Config.sample_rate

        # 2. Resample if necessary
        if sr != Config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=Config.sample_rate
            )
            waveform = resampler(waveform)

        # 3. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Adjust Duration (Pad or Crop)
        waveform = self._adjust_length(waveform)

        # 5. Feature Extraction (Log-Mel Spectrogram)
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 6. Normalization (Instance-based)
        # Normalize per sample to mean=0, std=1
        mean = spec.mean()
        std = spec.std()
        if std > 1e-6:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        # 7. Augmentation (Training only)
        if self.mode == "train":
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # 8. Prepare Label Vector
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        # Check if labels exist (Test set might not have labels)
        if "labels" in row and pd.notna(row["labels"]):
            # Labels are comma-separated strings
            label_list = str(row["labels"]).split(",")
            for lbl in label_list:
                lbl = lbl.strip()
                if lbl in self.class_to_idx:
                    label_vec[self.class_to_idx[lbl]] = 1.0

        return spec, label_vec

    def _adjust_length(self, waveform):
        """
        Pads or crops the waveform to the target length (30s).
        """
        # waveform shape: (channels, time)
        channels, time = waveform.shape
        target = Config.target_length

        if time < target:
            # Pad with zeros at the end
            padding = target - time
            waveform = F.pad(waveform, (0, padding))
        elif time > target:
            # Crop
            if self.mode == "train":
                # Random crop for training to expose different parts of the audio
                start = random.randint(0, time - target)
                waveform = waveform[:, start : start + target]
            else:
                # Truncate (take the beginning) for validation/test
                waveform = waveform[:, :target]

        return waveform
