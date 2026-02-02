import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
import torchvision.transforms as T
from library.config import Config
from library.utils import seed_everything


def load_or_generate_data(load_cached_data=True):
    """
    Loads spectrograms from cache or generates them from raw audio.
    Returns a dictionary mapping file_path (relative) to numpy array spectrograms.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "spectrograms.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print("Generating spectrograms from scratch...")
    # Gather all files from metadata
    dfs = []
    for p in [Config.TRAIN_CSV, Config.VAL_CSV, Config.TEST_CSV]:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise FileNotFoundError("No metadata files found.")

    full_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["file_path"])

    data_dict = {}

    # Define MelSpectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )

    # Amplitude to DB
    db_transform = torchaudio.transforms.AmplitudeToDB()

    for idx, row in full_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            # Load audio
            # Using soundfile for robustness, then converting to torch tensor
            wav, sr = sf.read(full_path)

            # Handle multi-channel (should be mono)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)

            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # (1, time)

            # Resample if necessary
            if sr != Config.SR:
                resampler = torchaudio.transforms.Resample(sr, Config.SR)
                wav_tensor = resampler(wav_tensor)

            # Pad/Crop to target duration
            target_len = Config.SR * Config.DURATION
            current_len = wav_tensor.shape[1]

            if current_len < target_len:
                pad = target_len - current_len
                wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad))
            elif current_len > target_len:
                wav_tensor = wav_tensor[:, :target_len]

            # Generate Spec
            spec = mel_transform(wav_tensor)
            spec = db_transform(spec)

            # Convert to numpy for storage (save memory/compatibility)
            # Shape: (1, n_mels, time) -> (n_mels, time)
            data_dict[rel_path] = spec.squeeze(0).numpy()

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data_dict)

    return data_dict


class BirdDataset(Dataset):
    def __init__(self, df, data_dict, augment=False):
        self.df = df.reset_index(drop=True)
        self.data_dict = data_dict
        self.augment = augment
        self.num_classes = Config.NUM_CLASSES
        # Lesson 00007: Use RandomAffine for time-shifting (horizontal translation)
        self.affine = T.RandomAffine(degrees=0, translate=(0.2, 0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Load spec
        if file_path in self.data_dict:
            spec = self.data_dict[file_path]  # (n_mels, time)
        else:
            spec = np.zeros((Config.N_MELS, 448), dtype=np.float32)

        spec_tensor = torch.from_numpy(spec)

        # Normalize [0, 1] per instance
        min_val = spec_tensor.min()
        max_val = spec_tensor.max()
        if max_val - min_val > 1e-6:
            spec_tensor = (spec_tensor - min_val) / (max_val - min_val)
        else:
            spec_tensor = torch.zeros_like(spec_tensor)

        # Resize entire spectrogram to (224, 224) - Lesson 00019
        # Input to interpolate must be (Batch, Channels, H, W) -> (1, 1, H, W)
        img_in = spec_tensor.unsqueeze(0).unsqueeze(0)
        img = (
            torch.nn.functional.interpolate(
                img_in, size=Config.IMG_SIZE, mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
        )  # (1, 224, 224)

        # Augmentation
        if self.augment:
            # Horizontal Translation (Time Shift) using RandomAffine - Lesson 00006/00007
            if torch.rand(1) < 0.5:
                img = self.affine(img)

            # Brightness Jitter - Lesson 00009
            if torch.rand(1) < 0.5:
                img = img + (torch.rand(1) * 0.2 - 0.1)

            # Contrast Jitter - Lesson 00009
            if torch.rand(1) < 0.5:
                factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
                mean = img.mean()
                img = (img - mean) * factor + mean

        # Replicate to 3 channels for ResNet - Lesson 00014
        img = img.repeat(3, 1, 1)  # (3, 224, 224)

        # Parse Labels
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        label_str = str(row["labels"])
        if label_str != "?" and label_str != "nan" and label_str.strip():
            try:
                indices = [int(x) for x in label_str.split()]
                label_vec[indices] = 1.0
            except ValueError:
                pass

        return img, torch.tensor(label_vec)


def get_dataloaders(fold=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold using Iterative Stratification.
    """
    # Load Data Dict
    data_dict = load_or_generate_data(load_cached_data=load_cached_data)

    # Load Metadata
    train_df_orig = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Use only the training set for Cross-Validation splits
    full_train_df = train_df_orig

    # Manually shuffle the dataframe to ensure random folds, as IterativeStratification
    # in skmultilearn 0.2.0 does not support passing random_state with shuffle=False
    full_train_df = full_train_df.sample(frac=1, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Prepare for Iterative Stratification
    X = full_train_df.index.values.reshape(-1, 1)

    # Create label matrix for stratification
    y = np.zeros((len(full_train_df), Config.NUM_CLASSES))
    for idx, row in full_train_df.iterrows():
        lbls = str(row["labels"])
        if lbls != "?" and lbls != "nan" and lbls.strip():
            try:
                for l in lbls.split():
                    y[idx, int(l)] = 1
            except ValueError:
                pass

    # Split
    # Use IterativeStratification to handle multi-label imbalance
    stratifier = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    # stratifier.split returns generator of indices
    splits = list(stratifier.split(X, y))

    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {len(splits)} splits.")

    train_indices, val_indices = splits[fold]

    train_subset = full_train_df.iloc[train_indices]
    val_subset = full_train_df.iloc[val_indices]

    # Debug Mode
    if Config.DEBUG:
        train_subset = train_subset.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_subset = val_subset.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Create Datasets
    train_ds = BirdDataset(train_subset, data_dict, augment=True)
    val_ds = BirdDataset(val_subset, data_dict, augment=False)
    test_ds = BirdDataset(test_df, data_dict, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
