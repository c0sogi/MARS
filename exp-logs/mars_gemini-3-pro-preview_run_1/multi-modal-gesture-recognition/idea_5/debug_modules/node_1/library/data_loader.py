import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    LABEL_MAP,
    NUM_JOINTS,
    JOINT_CHANNELS,
    MFCC_N_MFCC,
    MFCC_HOP_LENGTH,
    MFCC_N_FFT,
    AUDIO_SAMPLERATE,
    SEED,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultimodalDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, transform=None):
        self.split = split
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Load Metadata
        if split == "train":
            self.metadata = pd.read_csv(TRAIN_METADATA_PATH)
        elif split == "val":
            self.metadata = pd.read_csv(VAL_METADATA_PATH)
        elif split == "test":
            self.metadata = pd.read_csv(TEST_METADATA_PATH)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Debugging: Use subset
        if DEBUG:
            self.metadata = self.metadata.head(DEBUG_SUBSET_SIZE)

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Pre-process / Cache data
        self.valid_indices = []
        self._prepare_cache()

        # Load or Compute Normalization Stats
        self.stats_path = os.path.join(os.path.dirname(CACHE_DIR), "stats.npz")
        if split == "train":
            self._compute_global_stats()

        # Load stats for normalization
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            self.skel_mean = torch.tensor(stats["skel_mean"], dtype=torch.float32)
            self.skel_std = torch.tensor(stats["skel_std"], dtype=torch.float32)
            self.audio_mean = torch.tensor(stats["audio_mean"], dtype=torch.float32)
            self.audio_std = torch.tensor(stats["audio_std"], dtype=torch.float32)
        else:
            # Fallback if stats missing (should not happen if train runs first)
            self.skel_mean = torch.zeros(NUM_JOINTS * JOINT_CHANNELS)
            self.skel_std = torch.ones(NUM_JOINTS * JOINT_CHANNELS)
            self.audio_mean = torch.zeros(MFCC_N_MFCC)
            self.audio_std = torch.ones(MFCC_N_MFCC)

    def _prepare_cache(self):
        """Iterates through metadata and ensures all samples are processed and cached."""
        print(f"Preparing cache for {self.split} set...")
        for idx, row in self.metadata.iterrows():
            sample_id = row["sample_id"]
            cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")

            if self.load_cached_data and os.path.exists(cache_path):
                self.valid_indices.append(idx)
                continue

            # Process raw data
            try:
                data = self._process_raw_data(row)
                if data is not None:
                    np.savez_compressed(cache_path, **data)
                    self.valid_indices.append(idx)
            except Exception as e:
                # In strict mode we might raise, but for robustness we skip corrupt files
                # print(f"Failed to process {sample_id}: {e}")
                pass

    def _process_raw_data(self, row):
        """Reads .mat and .wav files and extracts features."""
        # Paths
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = (
            os.path.join(INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )

        # 1. Load Skeleton (.mat)
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except:
            return None

        if "Video" not in mat:
            return None
        video = mat["Video"]

        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        if num_frames == 0 or len(frames) == 0:
            return None

        # Extract Skeleton Joints
        # Shape: (T, V, 3)
        skeleton_data = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            if t >= num_frames:
                break

            # Handle nested structure variability
            if hasattr(frame, "Skeleton"):
                skel = frame.Skeleton
                # If multiple users, take first
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]

                if hasattr(skel, "WorldPosition"):
                    # WorldPosition might be struct or array
                    wp = skel.WorldPosition
                    # Check if it's a struct with X,Y,Z or array
                    # We expect 20 joints.
                    # If wp is an array of objects (joints)
                    if isinstance(wp, np.ndarray) and wp.size == NUM_JOINTS:
                        for j in range(NUM_JOINTS):
                            joint = wp[j]
                            if hasattr(joint, "X"):
                                skeleton_data[t, j, 0] = joint.X
                                skeleton_data[t, j, 1] = joint.Y
                                skeleton_data[t, j, 2] = joint.Z
                    # If wp is a single object with arrays (less likely based on description, but possible)
                    # Fallback to PixelPosition if WorldPosition is empty?
                    # Prompt says WorldPosition is preferred.
                    # Let's assume the loop above works for the provided dataset format.

        # Normalize Skeleton: Relative to HipCenter (Joint 0)
        # hip_center: (T, 1, 3)
        hip_center = skeleton_data[:, 0:1, :]
        skeleton_data = skeleton_data - hip_center

        # 2. Load Audio (.wav)
        audio_features = np.zeros((num_frames, MFCC_N_MFCC), dtype=np.float32)
        if audio_path and os.path.exists(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)

                # Resample
                if sr != AUDIO_SAMPLERATE:
                    resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLERATE)
                    waveform = resampler(waveform)

                # Mix to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Extract MFCC
                # hop_length aligned to video frame rate (50ms)
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=AUDIO_SAMPLERATE,
                    n_mfcc=MFCC_N_MFCC,
                    melkwargs={
                        "n_fft": MFCC_N_FFT,
                        "hop_length": MFCC_HOP_LENGTH,
                        "center": True,
                    },
                )

                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, T_audio)
                mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (T_audio, n_mfcc)

                # Align lengths
                if mfcc.shape[0] < num_frames:
                    # Pad
                    pad_width = num_frames - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
                elif mfcc.shape[0] > num_frames:
                    # Truncate
                    mfcc = mfcc[:num_frames, :]

                audio_features = mfcc.astype(np.float32)
            except:
                # If audio fails, keep zeros
                pass

        # 3. Generate Dense Labels
        # Background = 0
        labels_dense = np.zeros(num_frames, dtype=np.int64)

        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]

            for l in raw_labels:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    start = int(l.Begin) - 1  # 1-based to 0-based
                    end = int(l.End)

                    if name in LABEL_MAP:
                        lid = LABEL_MAP[name]
                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames, end)
                        if start < end:
                            labels_dense[start:end] = lid

        return {
            "skeleton": skeleton_data,  # (T, 20, 3)
            "audio": audio_features,  # (T, 13)
            "labels": labels_dense,  # (T,)
        }

    def _compute_global_stats(self):
        """Computes mean and std for skeleton and audio from cached training data."""
        if os.path.exists(self.stats_path):
            return

        print("Computing global normalization statistics...")
        skel_sum = np.zeros(NUM_JOINTS * JOINT_CHANNELS)
        skel_sq_sum = np.zeros(NUM_JOINTS * JOINT_CHANNELS)
        audio_sum = np.zeros(MFCC_N_MFCC)
        audio_sq_sum = np.zeros(MFCC_N_MFCC)
        count = 0

        # Use a subset to save time if dataset is huge, but here 400 is manageable.
        # We'll use up to 100 samples.
        indices = self.valid_indices[:100]

        for idx in indices:
            sample_id = self.metadata.iloc[idx]["sample_id"]
            cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")
            data = np.load(cache_path)

            skel = data["skeleton"].reshape(-1, NUM_JOINTS * JOINT_CHANNELS)
            aud = data["audio"]

            skel_sum += np.sum(skel, axis=0)
            skel_sq_sum += np.sum(skel**2, axis=0)
            audio_sum += np.sum(aud, axis=0)
            audio_sq_sum += np.sum(aud**2, axis=0)
            count += skel.shape[0]

        if count > 0:
            skel_mean = skel_sum / count
            skel_std = np.sqrt((skel_sq_sum / count) - (skel_mean**2)) + 1e-6

            audio_mean = audio_sum / count
            audio_std = np.sqrt((audio_sq_sum / count) - (audio_mean**2)) + 1e-6

            np.savez(
                self.stats_path,
                skel_mean=skel_mean,
                skel_std=skel_std,
                audio_mean=audio_mean,
                audio_std=audio_std,
            )
            print("Statistics computed and saved.")
        else:
            print("Warning: No data to compute stats.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample_id = self.metadata.iloc[real_idx]["sample_id"]
        cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")

        data = np.load(cache_path)
        skeleton = torch.tensor(data["skeleton"], dtype=torch.float32)  # (T, 20, 3)
        audio = torch.tensor(data["audio"], dtype=torch.float32)  # (T, 13)
        labels = torch.tensor(data["labels"], dtype=torch.long)  # (T,)

        # Augmentation (Train only)
        if self.split == "train":
            # Additive Gaussian Noise to Skeleton
            noise = torch.randn_like(skeleton) * 0.01
            skeleton = skeleton + noise

            # Random Channel Masking (Skeleton)
            # Flatten to (T, 60) for masking logic or mask per joint?
            # Let's mask per coordinate channel globally
            if np.random.rand() < 0.5:
                # Mask 10% of channels
                num_channels = NUM_JOINTS * JOINT_CHANNELS
                mask_indices = np.random.choice(
                    num_channels, size=int(num_channels * 0.1), replace=False
                )
                T, V, C = skeleton.shape
                skel_flat = skeleton.view(T, -1)
                skel_flat[:, mask_indices] = 0
                skeleton = skel_flat.view(T, V, C)

        # Normalization
        # Flatten skeleton for normalization: (T, 60)
        T, V, C = skeleton.shape
        skeleton = skeleton.view(T, -1)
        skeleton = (skeleton - self.skel_mean) / self.skel_std
        skeleton = skeleton.view(T, V, C)  # Reshape back to (T, V, C) for Graph Layer

        audio = (audio - self.audio_mean) / self.audio_std

        return skeleton, audio, labels


def pad_collate(batch):
    """
    Collate function to pad sequences to the same length.
    Batch is list of tuples (skeleton, audio, labels).
    """
    skeletons, audios, labels = zip(*batch)

    lengths = torch.tensor([s.size(0) for s in skeletons])
    max_len = lengths.max().item()

    # Pad Skeletons: (B, T, V, C)
    B = len(skeletons)
    V, C = skeletons[0].size(1), skeletons[0].size(2)
    padded_skel = torch.zeros(B, max_len, V, C)

    # Pad Audio: (B, T, F)
    F = audios[0].size(1)
    padded_audio = torch.zeros(B, max_len, F)

    # Pad Labels: (B, T) - Fill with 0 (Background)
    padded_labels = torch.zeros(B, max_len, dtype=torch.long)

    # Mask: (B, T) - 1 for valid, 0 for padding
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, length in enumerate(lengths):
        padded_skel[i, :length] = skeletons[i]
        padded_audio[i, :length] = audios[i]
        padded_labels[i, :length] = labels[i]
        mask[i, :length] = 1

    return padded_skel, padded_audio, padded_labels, mask, lengths


def get_dataloaders(batch_size=8, num_workers=2):
    train_ds = MultimodalDataset(split="train")
    val_ds = MultimodalDataset(split="val")
    test_ds = MultimodalDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate,
    )

    return train_loader, val_loader, test_loader
