import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_mat_polymorphic, compute_stats


def prepare_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads and preprocesses data for a given split.
    Implements caching to avoid re-parsing .mat and .wav files.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split_name}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Allow pickle is needed because we store a list of dictionaries (objects)
            data = np.load(cache_path, allow_pickle=True)
            print(f"Loaded cached {split_name} data from {cache_path}")
            return data["samples"]
        except Exception as e:
            print(f"Failed to load cache for {split_name}: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {split_name} data from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    samples = []

    # Audio Transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=Config.AUDIO_MFCC_N_MFCC
    )

    for idx, row in df.iterrows():
        # --- Load Skeleton ---
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        skeleton = load_mat_polymorphic(mat_path)  # Shape: (T, 20, 3)

        if skeleton is None:
            continue

        T = skeleton.shape[0]
        if T < 2:
            continue  # Need at least 2 frames for velocity

        # --- Load Audio ---
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        mfcc_features = np.zeros((T, Config.AUDIO_INPUT_DIM), dtype=np.float32)

        if os.path.exists(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)

                # Resample to 16k
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                # Convert to Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Compute MFCC
                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time_audio)

                # Resize to match Video Frames (T)
                # Input to interpolate must be (Batch, Channels, Time)
                if mfcc.shape[2] > 0:
                    mfcc_resized = torch.nn.functional.interpolate(
                        mfcc, size=T, mode="linear", align_corners=False
                    )
                    # Transpose to (T, n_mfcc)
                    mfcc_features = mfcc_resized.squeeze(0).transpose(0, 1).numpy()

            except Exception:
                # Fallback to zeros if audio fails
                pass

        # --- Create Labels ---
        labels = np.zeros(T, dtype=np.int64)  # Default 0 (Background)

        # Parse labels if they exist (Train/Val)
        if "labels" in row and isinstance(row["labels"], str):
            try:
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # Parse 1-based indices from metadata
                    start = max(0, int(l["begin"]) - 1)
                    end = min(T, int(l["end"]))
                    gid = int(l["id"])

                    if gid in Config.LABEL_MAP.values():
                        labels[start:end] = gid
            except json.JSONDecodeError:
                pass

        samples.append(
            {
                "sample_id": row["sample_id"],
                "skeleton": skeleton.astype(np.float32),
                "audio": mfcc_features.astype(np.float32),
                "labels": labels,
            }
        )

    # 3. Save Cache
    # Use object array to handle variable length sequences
    samples_array = np.array(samples, dtype=object)
    np.savez_compressed(cache_path, samples=samples_array)
    print(f"Saved {split_name} data to {cache_path}")

    return samples_array


class GestureDataset(Dataset):
    def __init__(self, split, load_cached_data=True, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached data.
            debug (bool): If True, limits dataset size.
        """
        self.split = split
        self.is_train = split == "train"

        # Determine Metadata Path
        if split == "train":
            meta_path = Config.TRAIN_METADATA
        elif split == "val":
            meta_path = Config.VAL_METADATA
        else:
            meta_path = Config.TEST_METADATA

        # Load Data
        self.samples = prepare_data(meta_path, split, load_cached_data)

        # Debug Mode
        if debug:
            self.samples = self.samples[: Config.DEBUG_SAMPLE_SIZE]

        # Load Normalization Stats
        stats = compute_stats(load_cached_data=load_cached_data)
        self.skel_mean = torch.tensor(stats["skeleton_mean"], dtype=torch.float32)
        self.skel_std = torch.tensor(stats["skeleton_std"], dtype=torch.float32)
        self.audio_mean = torch.tensor(stats["audio_mean"], dtype=torch.float32)
        self.audio_std = torch.tensor(stats["audio_std"], dtype=torch.float32)

        # Pre-calculate Sliding Windows for Training
        self.windows = []
        if self.is_train:
            stride = Config.STRIDE_TRAIN
            w_size = Config.WINDOW_SIZE

            for idx, sample in enumerate(self.samples):
                T = sample["skeleton"].shape[0]

                # If sequence is shorter than window, take one window (will be padded)
                if T <= w_size:
                    self.windows.append((idx, 0))
                else:
                    # Generate windows
                    for start in range(0, T - w_size + 1, stride):
                        self.windows.append((idx, start))

                    # Ensure coverage of the end
                    if (T - w_size) % stride != 0:
                        self.windows.append((idx, T - w_size))

    def __len__(self):
        if self.is_train:
            return len(self.windows)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            return self._getitem_train(idx)
        else:
            return self._getitem_val_test(idx)

    def _getitem_train(self, idx):
        sample_idx, start_frame = self.windows[idx]
        sample = self.samples[sample_idx]

        skel = sample["skeleton"]  # (T, 20, 3)
        audio = sample["audio"]  # (T, 13)
        lbls = sample["labels"]  # (T,)

        T = skel.shape[0]
        end_frame = start_frame + Config.WINDOW_SIZE

        # Handle Padding if sequence < window_size
        if T < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - T
            # Pad Skeleton: Edge padding to maintain posture
            skel_win = np.pad(skel, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            # Pad Audio: Constant 0
            audio_win = np.pad(audio, ((0, pad_len), (0, 0)), mode="constant")
            # Pad Labels: Constant 0 (Background)
            lbl_win = np.pad(lbls, (0, pad_len), mode="constant", constant_values=0)
        else:
            skel_win = skel[start_frame:end_frame]
            audio_win = audio[start_frame:end_frame]
            lbl_win = lbls[start_frame:end_frame]

        # --- Kinematically Consistent Augmentation ---
        # 1. Random Rotation (Y-axis)
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: Reshape to (N, 3) -> Rotate -> Reshape back
        skel_reshaped = skel_win.reshape(-1, 3)
        skel_aug = skel_reshaped @ R.T
        skel_win = skel_aug.reshape(Config.WINDOW_SIZE, 20, 3)

        # 2. Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        skel_win = skel_win * scale

        # --- Derive Velocity & Acceleration ---
        # Pad first frame to compute diff while keeping size
        skel_padded = np.pad(skel_win, ((1, 0), (0, 0), (0, 0)), mode="edge")
        vel = np.diff(skel_padded, axis=0)  # (W, 20, 3)

        vel_padded = np.pad(vel, ((1, 0), (0, 0), (0, 0)), mode="edge")
        acc = np.diff(vel_padded, axis=0)  # (W, 20, 3)

        # --- Flatten & Fuse ---
        p_flat = skel_win.reshape(Config.WINDOW_SIZE, -1)
        v_flat = vel.reshape(Config.WINDOW_SIZE, -1)
        a_flat = acc.reshape(Config.WINDOW_SIZE, -1)

        skel_feats = np.concatenate([p_flat, v_flat, a_flat], axis=1)  # (W, 180)

        # --- Normalization ---
        skel_feats = (skel_feats - self.skel_mean.numpy()) / self.skel_std.numpy()
        audio_feats = (audio_win - self.audio_mean.numpy()) / self.audio_std.numpy()

        # Handle NaNs
        skel_feats = np.nan_to_num(skel_feats)
        audio_feats = np.nan_to_num(audio_feats)

        # Final Fusion
        features = np.concatenate([skel_feats, audio_feats], axis=1)  # (W, 193)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(lbl_win, dtype=torch.long),
        }

    def _getitem_val_test(self, idx):
        sample = self.samples[idx]

        skel = sample["skeleton"]
        audio = sample["audio"]
        lbls = sample["labels"]

        T = skel.shape[0]

        # --- Derive Vel/Acc (No Augmentation) ---
        skel_padded = np.pad(skel, ((1, 0), (0, 0), (0, 0)), mode="edge")
        vel = np.diff(skel_padded, axis=0)

        vel_padded = np.pad(vel, ((1, 0), (0, 0), (0, 0)), mode="edge")
        acc = np.diff(vel_padded, axis=0)

        # --- Flatten ---
        p_flat = skel.reshape(T, -1)
        v_flat = vel.reshape(T, -1)
        a_flat = acc.reshape(T, -1)

        skel_feats = np.concatenate([p_flat, v_flat, a_flat], axis=1)

        # --- Normalization ---
        skel_feats = (skel_feats - self.skel_mean.numpy()) / self.skel_std.numpy()
        audio_feats = (audio - self.audio_mean.numpy()) / self.audio_std.numpy()

        skel_feats = np.nan_to_num(skel_feats)
        audio_feats = np.nan_to_num(audio_feats)

        features = np.concatenate([skel_feats, audio_feats], axis=1)

        return {
            "sample_id": sample["sample_id"],
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(lbls, dtype=torch.long),
        }


def get_dataloaders(batch_size=32, num_workers=4, debug=False):
    """
    Factory function to create dataloaders for all splits.
    """
    train_ds = GestureDataset("train", debug=debug)
    val_ds = GestureDataset("val", debug=debug)
    test_ds = GestureDataset("test", debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Batch size 1 for validation and test to handle variable sequence lengths
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
