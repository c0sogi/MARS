import os
import random
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    """
    Creates a Mel Filterbank matrix using NumPy.
    Returns a tensor of shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # Mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin points
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        if center != left:
            for j in range(left, center):
                fbank[i - 1, j] = (j - left) / (center - left)
        if right != center:
            for j in range(center, right):
                fbank[i - 1, j] = (right - j) / (right - center)

    return torch.tensor(fbank, dtype=torch.float32)


class AudioDataset(Dataset):
    def __init__(self, csv_file, phase="train", transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            phase (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.phase = phase
        self.transform = transform

        # Pre-compute Mel Filterbank
        self.mel_basis = create_mel_filterbank(
            Config.SAMPLE_RATE,
            Config.N_FFT,
            Config.N_MELS,
            Config.FMIN,
            Config.FMAX,
        )

        # Load class mapping
        # We derive classes from sample_submission.csv to ensure consistency with the competition format
        sample_sub_path = os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            ss_df = pd.read_csv(sample_sub_path, nrows=1)
            self.classes = [c for c in ss_df.columns if c not in ["fname", "file_path"]]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            # Fallback if file missing (unlikely given problem description)
            self.classes = []
            self.class_to_idx = {}

    def __len__(self):
        if Config.DEBUG:
            return min(len(self.df), Config.DEBUG_SUBSET_SIZE)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        try:
            # 1. Load Audio using soundfile
            # sf.read returns (data, sr). data is (frames, channels) or (frames,)
            waveform, sr = sf.read(file_path)

            # Convert to torch tensor: (Channels, Time)
            waveform = torch.from_numpy(waveform).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, Time)
            else:
                waveform = waveform.t()  # (Channels, Time)

            # 2. Resample if necessary
            if sr != Config.SAMPLE_RATE:
                # Use torch.nn.functional.interpolate for resampling
                # Input must be (Batch, Channels, Time)
                waveform = waveform.unsqueeze(0)
                waveform = torch.nn.functional.interpolate(
                    waveform,
                    scale_factor=Config.SAMPLE_RATE / sr,
                    mode="linear",
                    align_corners=False,
                )
                waveform = waveform.squeeze(0)

            # Ensure 1 channel (Mono)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Padding / Trimming
            target_length = int(Config.TRAIN_DURATION * Config.SAMPLE_RATE)
            current_length = waveform.shape[1]

            if self.phase == "train":
                if current_length > target_length:
                    # Random Crop
                    start = np.random.randint(0, current_length - target_length)
                    waveform = waveform[:, start : start + target_length]
                elif current_length < target_length:
                    # Pad
                    pad_amount = target_length - current_length
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            else:
                # Val/Test: No cropping, just pad if too short (optional, or handle in collate)
                pass

            # 4. Mel Spectrogram (Manual Implementation)
            # STFT
            window = torch.hann_window(Config.N_FFT)
            stft = torch.stft(
                waveform,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH,
                win_length=Config.N_FFT,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            # Power Spectrum: (Batch, Freq, Time)
            power_spec = stft.abs().pow(2.0)

            # Apply Mel Basis: (Mels, Freq) @ (Freq, Time) -> (Mels, Time)
            # waveform is (1, Time), power_spec is (1, Freq, Time)
            mel_spec = torch.matmul(self.mel_basis, power_spec)

            # 5. Log Scale (AmplitudeToDB)
            # 10 * log10(x + epsilon)
            spec = 10.0 * torch.log10(mel_spec + 1e-10)

            # 6. Instance-wise Normalization
            if Config.NORMALIZE_INSTANCE:
                mean = spec.mean()
                std = spec.std()
                spec = (spec - mean) / (std + 1e-6)

            # 7. Augmentation (Train only)
            if self.phase == "train" and Config.USE_SPEC_AUGMENT:
                # Frequency Masking
                f_mask_param = Config.SPEC_AUG_FREQ_MASK
                f_bank_size = spec.shape[1]  # (1, Mels, Time) -> Mels is dim 1
                f = np.random.randint(0, f_mask_param)
                f0 = np.random.randint(0, f_bank_size - f)
                spec[:, f0 : f0 + f, :] = spec.mean()  # Mask with mean or 0

                # Time Masking
                t_mask_param = Config.SPEC_AUG_TIME_MASK
                t_steps = spec.shape[2]
                t = np.random.randint(0, t_mask_param)
                t0 = np.random.randint(0, t_steps - t)
                spec[:, :, t0 : t0 + t] = spec.mean()

            # 8. Prepare Label
            label_vec = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            if self.phase in ["train", "val"]:
                labels_str = str(row["labels"])
                if pd.notna(labels_str) and labels_str != "":
                    labels = labels_str.split(",")
                    for l in labels:
                        l = l.strip()
                        if l in self.class_to_idx:
                            label_vec[self.class_to_idx[l]] = 1.0

            return spec, label_vec, row["fname"]

        except Exception as e:
            # Fallback for corrupted files
            print(f"Error loading {file_path}: {e}")
            dummy_spec = torch.zeros(
                (
                    1,
                    Config.N_MELS,
                    int(Config.TRAIN_DURATION * Config.SAMPLE_RATE // Config.HOP_LENGTH)
                    + 1,
                )
            )
            dummy_label = torch.zeros(Config.NUM_CLASSES)
            return dummy_spec, dummy_label, row["fname"]


def collate_fn(batch):
    """
    Collate function to handle variable length spectrograms.
    Pads the time dimension to the maximum length in the batch.
    """
    # batch is list of (spec, label, fname)
    # Filter out None/Errors if any
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), []

    specs, labels, fnames = zip(*batch)

    # specs are (1, n_mels, time)
    # Find max time in this batch
    max_time = max([s.shape[2] for s in specs])

    padded_specs = []
    for s in specs:
        current_time = s.shape[2]
        pad_amount = max_time - current_time
        if pad_amount > 0:
            # Pad last dim (time)
            s_padded = torch.nn.functional.pad(s, (0, pad_amount))
            padded_specs.append(s_padded)
        else:
            padded_specs.append(s)

    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.stack(labels)

    return specs_tensor, labels_tensor, list(fnames)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_dataloaders():
    """
    Creates dataloaders for train, val, and test.
    """
    train_dataset = AudioDataset(Config.TRAIN_CSV, phase="train")
    val_dataset = AudioDataset(Config.VAL_CSV, phase="val")
    test_dataset = AudioDataset(Config.TEST_CSV, phase="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
