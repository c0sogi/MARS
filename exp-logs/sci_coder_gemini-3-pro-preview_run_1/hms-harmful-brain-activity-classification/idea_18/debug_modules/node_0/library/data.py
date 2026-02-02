import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Standard 10-20 System + EKG
EEG_FEATURES = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]


def load_eeg(
    file_path, offset_seconds, duration_seconds=50, fs=200, target_fs=100, is_test=False
):
    """
    Loads raw EEG parquet, crops to window, downsamples, and applies instance norm.
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        # Return zeros if file load fails
        return np.zeros(
            (int(duration_seconds * target_fs), len(EEG_FEATURES)), dtype=np.float32
        )

    # Align columns
    available_cols = [c for c in EEG_FEATURES if c in df.columns]
    if len(available_cols) < len(EEG_FEATURES):
        # Pad missing columns with zeros
        data = np.zeros((len(df), len(EEG_FEATURES)), dtype=np.float32)
        for i, col in enumerate(EEG_FEATURES):
            if col in df.columns:
                data[:, i] = df[col].values
    else:
        data = df[available_cols].values

    # Crop Time Window
    if not is_test:
        start_idx = int(offset_seconds * fs)
        end_idx = start_idx + int(duration_seconds * fs)

        # Handle boundary conditions
        if start_idx < 0:
            start_idx = 0

        if end_idx > len(data):
            pad_len = end_idx - len(data)
            data = np.pad(data, ((0, pad_len), (0, 0)), "constant")
            data = data[start_idx:end_idx]
        else:
            data = data[start_idx:end_idx]

    # Handle NaNs
    data = np.nan_to_num(data, nan=0.0)

    # Downsample (e.g., 200Hz -> 100Hz)
    step = int(fs / target_fs)
    data = data[::step]

    # Ensure exact length
    target_len = int(duration_seconds * target_fs)
    if len(data) > target_len:
        data = data[:target_len]
    elif len(data) < target_len:
        data = np.pad(data, ((0, target_len - len(data)), (0, 0)), "constant")

    # Instance Normalization (Channel-wise)
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    data = (data - mean) / (std + 1e-6)

    # Transpose to (Channels, Time) for 1D Conv
    data = data.transpose(1, 0)

    return data.astype(np.float32)


def load_spectrogram(file_path, offset_seconds, spec_size=(512, 512), is_test=False):
    """
    Loads spectrogram, crops 10m window, resizes, and adds Coordinate Map channel.
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        return np.zeros((spec_size[0], spec_size[1], 5), dtype=np.float32)

    if is_test:
        spec_data = df.values
    else:
        # Crop 10 minutes (600s) based on offset
        # Assuming parquet rows correspond to time.
        # If 'time' column exists, use it. Else assume full file is relevant or handle via index.
        if "time" in df.columns:
            mask = (df["time"] >= offset_seconds) & (df["time"] < offset_seconds + 600)
            spec_data = df.loc[mask].drop(columns=["time"], errors="ignore").values
        else:
            # Fallback: Just take the values.
            # In this dataset, train_spectrograms often contain the full recording.
            # Without 'time' column, precise cropping is hard, but usually 'time' is present.
            # If not, we use the whole file (often pre-cropped in some versions of data).
            spec_data = df.values

    # Handle NaNs and Log Transform
    spec_data = np.nan_to_num(spec_data, nan=0.0)
    spec_data = np.log1p(spec_data)

    # Resize to target resolution (H, W)
    # We need to split into 4 regions (LL, RL, LP, RP)
    # Heuristic: Split width into 4 chunks
    h, w = spec_data.shape
    if w >= 4:
        w_chunk = w // 4
        c1 = cv2.resize(spec_data[:, 0:w_chunk], spec_size)
        c2 = cv2.resize(spec_data[:, w_chunk : 2 * w_chunk], spec_size)
        c3 = cv2.resize(spec_data[:, 2 * w_chunk : 3 * w_chunk], spec_size)
        c4 = cv2.resize(spec_data[:, 3 * w_chunk :], spec_size)
        img_stack = np.stack([c1, c2, c3, c4], axis=-1)
    else:
        # Fallback for weird shapes
        resized = cv2.resize(spec_data, spec_size)
        img_stack = np.stack([resized] * 4, axis=-1)

    # Generate Coordinate Map (Channel 5)
    # Linear gradient along the Time axis (Height) representing relative temporal distance
    # [-1, 1]
    time_steps = spec_size[0]
    coord_map = np.linspace(-1, 1, time_steps).astype(np.float32)
    # Expand to (H, W)
    coord_map = np.tile(coord_map[:, None], (1, spec_size[1]))
    coord_map = coord_map[..., None]  # (H, W, 1)

    # Concatenate: (H, W, 4) + (H, W, 1) -> (H, W, 5)
    final_img = np.concatenate([img_stack, coord_map], axis=-1)

    return final_img


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.XYMasking(
                    num_masks_x=(1, 2),
                    mask_x_length=(10, 20),
                    num_masks_y=(1, 2),
                    mask_y_length=(10, 20),
                    p=0.5,
                ),
                ToTensorV2(transpose_mask=True),  # (H, W, C) -> (C, H, W)
            ]
        )
    else:
        return A.Compose([ToTensorV2(transpose_mask=True)])


class EEGDataset(Dataset):
    def __init__(self, data_list, config, mode="train", augment=True):
        self.data_list = data_list
        self.config = config
        self.mode = mode
        self.augment = augment
        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Load cached tensors
        try:
            eeg = np.load(item["eeg_npy_path"])
            spec = np.load(item["spec_npy_path"])
        except Exception:
            # Silent fallback
            eeg = np.zeros(
                (self.config.EEG_CHANNELS, self.config.EEG_SEQ_LEN), dtype=np.float32
            )
            spec = np.zeros(
                (
                    self.config.SPEC_SIZE[0],
                    self.config.SPEC_SIZE[1],
                    self.config.SPEC_CHANNELS,
                ),
                dtype=np.float32,
            )

        # Augment EEG: Channel Dropout
        if self.mode == "train" and self.augment:
            if np.random.rand() < 0.5:
                ch = np.random.randint(0, eeg.shape[0])
                eeg[ch, :] = 0.0

        # Augment Spectrogram: Albumentations
        # Input spec is (H, W, 5)
        augmented = self.transforms(image=spec)
        spec_tensor = augmented["image"]  # (5, H, W)

        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

        if self.mode in ["train", "val"]:
            target = torch.tensor(item["target"], dtype=torch.float32)
            return eeg_tensor, spec_tensor, target
        else:
            return eeg_tensor, spec_tensor


def process_and_cache_dataset(df, config, mode="train"):
    """
    Iterates dataframe, processes raw data, saves to cache, returns list of cache paths.
    """
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    processed_list = []
    print(f"Preparing {len(df)} samples for {mode} (Caching to {cache_dir})...")

    for idx, row in df.iterrows():
        eeg_id = row["eeg_id"]

        if mode == "test":
            uid = f"{eeg_id}"
            offset_eeg = 0
            offset_spec = 0
        else:
            # Use sub_id for uniqueness in train
            sub_id = row.get("eeg_sub_id", idx)
            uid = f"{eeg_id}_{sub_id}"
            offset_eeg = row["eeg_label_offset_seconds"]
            offset_spec = row["spectrogram_label_offset_seconds"]

        eeg_npy_path = os.path.join(cache_dir, f"eeg_{uid}.npy")
        spec_npy_path = os.path.join(cache_dir, f"spec_{uid}.npy")

        # Process if not cached
        if not (os.path.exists(eeg_npy_path) and os.path.exists(spec_npy_path)):
            eeg_path = os.path.join(config.INPUT_DIR, row["eeg_path"])
            spec_path = os.path.join(config.INPUT_DIR, row["spectrogram_path"])

            eeg_data = load_eeg(
                eeg_path,
                offset_eeg,
                duration_seconds=config.EEG_DURATION,
                fs=200,
                target_fs=config.EEG_SR,
                is_test=(mode == "test"),
            )

            spec_data = load_spectrogram(
                spec_path,
                offset_spec,
                spec_size=config.SPEC_SIZE,
                is_test=(mode == "test"),
            )

            np.save(eeg_npy_path, eeg_data)
            np.save(spec_npy_path, spec_data)

        item = {
            "eeg_npy_path": eeg_npy_path,
            "spec_npy_path": spec_npy_path,
            "eeg_id": eeg_id,
        }

        if mode in ["train", "val"]:
            # Map vote columns to probabilities
            # Column names in df are like 'seizure_prob'
            probs = []
            for c in config.CLASS_NAMES:
                # c is 'seizure_vote', we need 'seizure_prob'
                prob_col = c.replace("_vote", "_prob")
                probs.append(row[prob_col])
            item["target"] = np.array(probs, dtype=np.float32)

        processed_list.append(item)

    return processed_list


def get_dataloader(df, config, mode="train", batch_size=32, shuffle=True):
    # Global Random Subsampling for Training
    if mode == "train" and config.USE_SUBSET:
        if len(df) > config.SUBSET_SIZE:
            df = df.sample(n=config.SUBSET_SIZE, random_state=config.SEED).reset_index(
                drop=True
            )
            print(f"Subsampled training set to {len(df)} samples.")

    # Pre-process and Cache
    data_list = process_and_cache_dataset(df, config, mode)

    # Create Dataset
    ds = EEGDataset(data_list, config, mode=mode, augment=(mode == "train"))

    # Create DataLoader
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=(mode == "train"),
    )

    return dl
