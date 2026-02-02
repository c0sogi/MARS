import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import CFG
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(CFG.seed)


def compute_spectrogram(file_path):
    """
    Computes Log-Mel Spectrogram from a wav file.
    """
    try:
        # Construct full path
        full_path = os.path.join(CFG.input_root, file_path)

        # Load audio
        wav, sr = sf.read(full_path)

        # Pad or crop to target duration
        target_length = CFG.duration * CFG.sr
        if len(wav) < target_length:
            padding = target_length - len(wav)
            wav = np.pad(wav, (0, padding), mode="constant")
        else:
            wav = wav[:target_length]

        # Convert to tensor
        wav_tensor = torch.from_numpy(wav).float()

        # Compute Mel Spectrogram
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.sr,
            n_mels=CFG.n_mels,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
        )

        mel_spec = transform(wav_tensor)

        # Convert to Log-Mel (Power -> dB)
        # Add small epsilon to avoid log(0)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        return log_mel_spec

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero tensor of expected shape in case of failure
        # Expected time steps: ceil(target_length / hop_length) + 1 approx
        n_frames = int(target_length / CFG.hop_length) + 1
        return torch.zeros((CFG.n_mels, n_frames))


def load_or_compute_spectrograms(dfs, load_cached_data=True):
    """
    Loads spectrograms from cache or computes them from scratch.

    Args:
        dfs (list of pd.DataFrame): List of dataframes (train, val, test) to process.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of rec_id (int) -> spectrogram (torch.Tensor)
    """
    cache_dir = os.path.join(CFG.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "spectrograms.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached spectrograms from {cache_path}...")
            # Allow pickle is required for loading object arrays/dicts via npy
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing spectrograms from scratch...")
    data_dict = {}

    # Combine all unique files from provided dataframes
    all_df = pd.concat(dfs, ignore_index=True)
    unique_files = all_df[["rec_id", "file_path"]].drop_duplicates()

    total = len(unique_files)
    for idx, row in unique_files.iterrows():
        rec_id = row["rec_id"]
        file_path = row["file_path"]

        spec = compute_spectrogram(file_path)
        data_dict[rec_id] = spec.numpy()  # Store as numpy to save space/compatibility

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} files")

    # 3. Save to cache
    print(f"Saving spectrograms to {cache_path}...")
    np.save(cache_path, data_dict)

    return data_dict


class BirdDataset(Dataset):
    def __init__(self, df, spec_cache, phase="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            spec_cache (dict): Dictionary mapping rec_id to spectrogram numpy arrays.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.spec_cache = spec_cache
        self.phase = phase
        self.num_classes = CFG.num_classes

        # Pre-instantiate augmentations
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Retrieve spectrogram
        if rec_id in self.spec_cache:
            spec = torch.from_numpy(self.spec_cache[rec_id])
        else:
            # Fallback if id missing from cache (should not happen)
            spec = compute_spectrogram(row["file_path"])

        # Augmentations (Train only)
        if self.phase == "train":
            # 1. Time Shift (Rolling)
            if CFG.time_shift and np.random.random() < 0.5:
                shift = np.random.randint(0, spec.shape[1])
                spec = torch.roll(spec, shifts=shift, dims=1)

            # 2. SpecAugment
            if CFG.spec_augment:
                # Apply masking
                if np.random.random() < 0.5:
                    spec = self.time_masking(spec)
                if np.random.random() < 0.5:
                    spec = self.freq_masking(spec)

            # 3. Photometric (Brightness/Contrast)
            # spec is log-mel, so adding constant is brightness, multiplying is contrast
            if np.random.random() < 0.5:
                # Contrast
                factor = np.random.uniform(0.8, 1.2)
                spec = spec * factor
                # Brightness
                bias = np.random.uniform(-0.5, 0.5)
                spec = spec + bias

        # Normalization (Standardize per sample)
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        # 3-Channel Rule: Replicate
        # Shape becomes [3, n_mels, time]
        image = spec.unsqueeze(0).expand(3, -1, -1)

        # Process Labels
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        if self.phase != "test":
            labels_str = str(row["labels"])
            if labels_str != "?" and labels_str.lower() != "nan" and labels_str.strip():
                try:
                    indices = [int(x) for x in labels_str.split()]
                    for cls_idx in indices:
                        if 0 <= cls_idx < self.num_classes:
                            label_vec[cls_idx] = 1.0
                except ValueError:
                    pass

        return image, label_vec, rec_id


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    """
    Applies Mixup to inputs and targets.
    Returns mixed inputs, pairs of targets, and lambda.
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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
