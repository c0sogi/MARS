import os
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    STATS_PATH,
    SEED,
    BATCH_SIZE,
    INPUT_DIM_AUDIO,
    BACKGROUND_LABEL,
    LABEL_MAP,
)
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(
        self, metadata_df, mode="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing file paths and metadata.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Ensure cache directory exists
        self.cache_dir = (
            os.path.join(CACHE_DIR, f"cache_{mode}")
            if mode != "train"
            else os.path.join(CACHE_DIR, "cache_train")
        )
        # We share cache for train/val if they come from same source, but here we separate to be safe
        # or use a common cache based on Sample ID. Let's use a common cache structure based on Sample ID.
        self.cache_dir = os.path.join(CACHE_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load or Compute Global Stats for Normalization
        self.stats = self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        """
        Loads global mean/std from STATS_PATH.
        If not found and mode is 'train', computes them from the current dataset and saves.
        If not found and mode is not 'train', raises an error (stats should be computed on train).
        """
        if os.path.exists(STATS_PATH):
            return np.load(STATS_PATH)

        if self.mode == "train":
            print("Computing global stats for normalization...")
            pos_sum, pos_sq_sum, pos_count = 0, 0, 0
            vel_sum, vel_sq_sum, vel_count = 0, 0, 0
            aud_sum, aud_sq_sum, aud_count = 0, 0, 0

            # Iterate over all samples to compute stats
            # We don't cache here to avoid double storage usage, just process on the fly
            for idx in range(len(self.metadata)):
                data = self._process_sample(
                    idx, use_cache=False
                )  # Force re-read to ensure raw data
                if data is None:
                    continue

                # Position
                p = data["pos"]
                pos_sum += np.sum(p, axis=0)
                pos_sq_sum += np.sum(p**2, axis=0)
                pos_count += p.shape[0]

                # Velocity
                v = data["vel"]
                vel_sum += np.sum(v, axis=0)
                vel_sq_sum += np.sum(v**2, axis=0)
                vel_count += v.shape[0]

                # Audio
                a = data["audio"]
                aud_sum += np.sum(a, axis=0)
                aud_sq_sum += np.sum(a**2, axis=0)
                aud_count += a.shape[0]

            stats = {
                "pos_mean": pos_sum / pos_count,
                "pos_std": np.sqrt(
                    (pos_sq_sum / pos_count) - (pos_sum / pos_count) ** 2 + 1e-6
                ),
                "vel_mean": vel_sum / vel_count,
                "vel_std": np.sqrt(
                    (vel_sq_sum / vel_count) - (vel_sum / vel_count) ** 2 + 1e-6
                ),
                "audio_mean": aud_sum / aud_count,
                "audio_std": np.sqrt(
                    (aud_sq_sum / aud_count) - (aud_sum / aud_count) ** 2 + 1e-6
                ),
            }
            np.savez(STATS_PATH, **stats)
            print(f"Stats saved to {STATS_PATH}")
            return stats
        else:
            # If validation/test runs before train (unlikely pipeline), we can't normalize properly
            # But usually train runs first. If file missing, we warn or return identity.
            # For this task, we assume train runs first or stats are provided.
            # Fallback to zeros/ones if missing (should not happen in proper flow)
            return None

    def _extract_skeleton(self, mat_path):
        """Parses .mat file for skeleton data."""
        try:
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            if "Video" not in mat:
                return None
            video = mat["Video"]
            frames = video.Frames

            # Handle cases where Frames might be empty or single object
            if not isinstance(frames, np.ndarray):
                frames = np.array([frames]) if frames is not None else np.array([])

            num_frames = len(frames)
            if num_frames == 0:
                return None

            # 20 joints, 3 coords
            skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

            for i in range(num_frames):
                skel = frames[i].Skeleton
                # If multiple users, take first
                if isinstance(skel, np.ndarray):
                    if skel.size > 0:
                        skel = skel[0]
                    else:
                        continue

                if hasattr(skel, "WorldPosition"):
                    wp = skel.WorldPosition
                    # wp might be struct with X,Y,Z
                    skeleton_data[i, :, 0] = wp.X
                    skeleton_data[i, :, 1] = wp.Y
                    skeleton_data[i, :, 2] = wp.Z

            return skeleton_data
        except Exception as e:
            # print(f"Error reading skeleton {mat_path}: {e}")
            return None

    def _extract_labels(self, mat_path, num_frames):
        """Parses .mat file for labels."""
        labels = np.full(num_frames, BACKGROUND_LABEL, dtype=np.int64)
        try:
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            video = mat["Video"]

            if not hasattr(video, "Labels"):
                return labels

            raw_labels = video.Labels
            # Normalize to list
            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]

            for l in raw_labels:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in LABEL_MAP:
                        lid = LABEL_MAP[name]
                        # Matlab 1-based indexing
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))
                        labels[start:end] = lid
            return labels
        except:
            return labels

    def _extract_audio(self, audio_path, target_frames):
        """Extracts MFCCs and aligns to video frames."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Compute MFCC
            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=INPUT_DIM_AUDIO,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = transform(waveform)  # (n_mfcc, time)

            # Interpolate to match video frames
            mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (T, n_mfcc)

            return mfcc.numpy()
        except Exception as e:
            # Return zeros if audio fails
            return np.zeros((target_frames, INPUT_DIM_AUDIO), dtype=np.float32)

    def _process_sample(self, idx, use_cache=True):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.npz")

        # 1. Try Load Cache
        if use_cache and self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return dict(data)
            except:
                pass  # Corrupt cache, recompute

        # 2. Compute
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )

        # Skeleton
        skel = self._extract_skeleton(mat_path)
        if skel is None:
            return None  # Skip broken samples

        num_frames = skel.shape[0]

        # Normalize Skeleton (Relative to HipCenter at index 0)
        # Shape (T, 20, 3)
        hip = skel[:, 0:1, :]  # (T, 1, 3)
        pos = skel - hip
        pos = pos.reshape(num_frames, -1)  # Flatten (T, 60)

        # Velocity
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # Audio
        if audio_path and os.path.exists(audio_path):
            audio = self._extract_audio(audio_path, num_frames)
        else:
            audio = np.zeros((num_frames, INPUT_DIM_AUDIO), dtype=np.float32)

        # Labels
        labels = self._extract_labels(mat_path, num_frames)

        # Boundary Labels (1 where label changes)
        boundaries = np.zeros(num_frames, dtype=np.float32)
        # Change points
        diff = labels[1:] != labels[:-1]
        boundaries[1:][diff] = 1.0

        data = {
            "pos": pos.astype(np.float32),
            "vel": vel.astype(np.float32),
            "audio": audio.astype(np.float32),
            "labels": labels.astype(np.int64),
            "boundaries": boundaries.astype(np.float32),
        }

        # 3. Save Cache
        if self.load_cached_data:
            np.savez_compressed(cache_path, **data)

        return data

    def augment_sample(self, pos, vel, audio):
        """Applies augmentation to features."""
        # 1. Channel Masking
        if np.random.rand() < 0.5:
            # Mask 10% of channels in pos/vel
            mask_idx = np.random.choice(
                pos.shape[1], int(pos.shape[1] * 0.1), replace=False
            )
            pos[:, mask_idx] = 0
            vel[:, mask_idx] = 0

        # 2. Gaussian Noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, pos.shape)
            pos += noise
            vel += noise  # Velocity noise correlated? Simplified: independent

        # 3. Temporal Cutout
        if np.random.rand() < 0.3:
            T = pos.shape[0]
            if T > 20:
                cut_len = np.random.randint(5, 15)
                start = np.random.randint(0, T - cut_len)
                pos[start : start + cut_len] = 0
                vel[start : start + cut_len] = 0
                audio[start : start + cut_len] = 0

        return pos, vel, audio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self._process_sample(idx)
        if data is None:
            # Fallback for broken sample: return next one or random
            return self.__getitem__((idx + 1) % len(self))

        pos = data["pos"]
        vel = data["vel"]
        audio = data["audio"]
        labels = data["labels"]
        boundaries = data["boundaries"]

        # Normalization
        if self.stats is not None:
            pos = (pos - self.stats["pos_mean"]) / (self.stats["pos_std"] + 1e-6)
            vel = (vel - self.stats["vel_mean"]) / (self.stats["vel_std"] + 1e-6)
            audio = (audio - self.stats["audio_mean"]) / (
                self.stats["audio_std"] + 1e-6
            )

        # Augmentation (Train only)
        if self.mode == "train":
            pos, vel, audio = self.augment_sample(pos, vel, audio)

        return {
            "pos": torch.tensor(pos, dtype=torch.float32),
            "vel": torch.tensor(vel, dtype=torch.float32),
            "audio": torch.tensor(audio, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "boundaries": torch.tensor(boundaries, dtype=torch.float32),
            "sample_id": self.metadata.iloc[idx]["sample_id"],
        }


def collate_fn(batch):
    # Sort by length for packing (optional but good practice)
    batch.sort(key=lambda x: x["pos"].shape[0], reverse=True)

    pos = [x["pos"] for x in batch]
    vel = [x["vel"] for x in batch]
    audio = [x["audio"] for x in batch]
    labels = [x["labels"] for x in batch]
    boundaries = [x["boundaries"] for x in batch]
    ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([x.shape[0] for x in pos], dtype=torch.long)

    # Pad
    pos_pad = pad_sequence(pos, batch_first=True, padding_value=0)
    vel_pad = pad_sequence(vel, batch_first=True, padding_value=0)
    audio_pad = pad_sequence(audio, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=BACKGROUND_LABEL)
    boundaries_pad = pad_sequence(boundaries, batch_first=True, padding_value=0)

    return {
        "pos": pos_pad,
        "vel": vel_pad,
        "audio": audio_pad,
        "labels": labels_pad,
        "boundaries": boundaries_pad,
        "lengths": lengths,
        "sample_ids": ids,
    }


def get_dataloaders():
    """Reads metadata and returns DataLoaders."""
    # Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Datasets
    train_ds = GestureDataset(train_df, mode="train")
    val_ds = GestureDataset(val_df, mode="val")
    test_ds = GestureDataset(test_df, mode="test")

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
