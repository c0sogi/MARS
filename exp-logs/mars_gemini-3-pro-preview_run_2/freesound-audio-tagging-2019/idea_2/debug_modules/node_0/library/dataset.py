import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


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

        # Audio Transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentations
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK
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

        # 1. Load Audio
        try:
            info = torchaudio.info(file_path)
            sr = info.sample_rate
            total_frames = info.num_frames

            # Calculate frames needed at original SR to produce TRAIN_DURATION at Config.SAMPLE_RATE
            target_resampled_frames = int(Config.TRAIN_DURATION * Config.SAMPLE_RATE)
            resample_ratio = Config.SAMPLE_RATE / sr
            frames_needed_orig = int(target_resampled_frames / resample_ratio)

            waveform = None

            if self.phase == "train":
                if total_frames > frames_needed_orig:
                    # Random Crop
                    diff = total_frames - frames_needed_orig
                    start_frame = np.random.randint(0, diff)
                    waveform, _ = torchaudio.load(
                        file_path,
                        frame_offset=start_frame,
                        num_frames=frames_needed_orig,
                    )
                else:
                    # Load full file, pad later
                    waveform, _ = torchaudio.load(file_path)
            else:
                # Val/Test: Load full file
                waveform, _ = torchaudio.load(file_path)

            # 2. Resample
            if sr != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=Config.SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # 3. Padding / Trimming
            # Ensure 1 channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if self.phase == "train":
                current_frames = waveform.shape[1]
                if current_frames < target_resampled_frames:
                    pad_amount = target_resampled_frames - current_frames
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                elif current_frames > target_resampled_frames:
                    waveform = waveform[:, :target_resampled_frames]

            # 4. Mel Spectrogram
            spec = self.mel_spec(waveform)

            # 5. Log Scale
            spec = self.amplitude_to_db(spec)

            # 6. Instance-wise Normalization
            if Config.NORMALIZE_INSTANCE:
                mean = spec.mean()
                std = spec.std()
                spec = (spec - mean) / (std + 1e-6)

            # 7. Augmentation (Train only)
            if self.phase == "train" and Config.USE_SPEC_AUGMENT:
                spec = self.freq_masking(spec)
                spec = self.time_masking(spec)

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
