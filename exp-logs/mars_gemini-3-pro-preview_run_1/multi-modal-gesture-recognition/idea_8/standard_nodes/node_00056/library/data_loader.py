import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from library.config import Config
from library.utils import set_seed


class GestureDataset(Dataset):
    def __init__(self, metadata_df, mode="train", stats=None, load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing file paths and metadata.
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
            stats (dict): Dictionary containing mean/std for normalization.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        # Filter out rows where data_path is missing (NaN) to prevent TypeError in os.path.join
        # Cite debug_lesson_5
        self.metadata = metadata_df[metadata_df["data_path"].notna()].reset_index(
            drop=True
        )
        self.mode = mode
        self.stats = stats
        self.load_cached_data = load_cached_data

        # Audio transformation
        # Cite solution_lesson_node_00054: Match hop_length to frame rate (16000Hz / 20FPS = 800)
        self.mfcc_transform = T.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.AUDIO_INPUT_DIM,
            melkwargs={"n_fft": 400, "hop_length": 800, "n_mels": 23, "center": False},
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]

        # Try to load from cache
        cache_path = os.path.join(Config.CACHE_DIR, f"{sample_id}.npz")

        data = None
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                pose = torch.from_numpy(data["pose"]).float()
                audio = torch.from_numpy(data["audio"]).float()
                labels = torch.from_numpy(data["labels"]).long()
            except Exception as e:
                # If cache is corrupted, fall back to processing
                print(f"Error loading cache for {sample_id}: {e}")
                data = None

        if data is None:
            # Process from raw files
            # Cite solution_lesson_node_00053: Removed velocity and boundaries
            pose, audio, labels = self._process_raw_data(row)

            # Save to cache
            if self.load_cached_data:
                np.savez_compressed(
                    cache_path,
                    pose=pose.numpy(),
                    audio=audio.numpy(),
                    labels=labels.numpy(),
                )

        # Apply Normalization
        if self.stats:
            pose = (pose - self.stats["pose_mean"]) / (self.stats["pose_std"] + 1e-6)
            audio = (audio - self.stats["audio_mean"]) / (
                self.stats["audio_std"] + 1e-6
            )

        # Apply Augmentation (only in train mode)
        if self.mode == "train":
            pose, audio = self._augment_data(pose, audio)

        return pose, audio, labels

    def _process_raw_data(self, row):
        """Reads .mat and .wav files and extracts features."""
        # 1. Load MAT file for Skeleton and Labels
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            video_struct = mat["Video"]
            num_frames = video_struct.NumFrames

            # --- Skeleton Processing ---
            # Extract frames. Structure: Video.Frames[i].Skeleton.WorldPosition
            # We assume single user (UserIndex 1) or take the first valid skeleton

            # Pre-allocate
            skeleton_data = np.zeros(
                (num_frames, Config.NUM_JOINTS, 3), dtype=np.float32
            )

            frames = video_struct.Frames
            # Handle case where Frames is a single object or empty
            if not isinstance(frames, np.ndarray):
                frames = np.array([frames]) if frames is not None else np.array([])

            actual_frames = min(len(frames), num_frames)

            for i in range(actual_frames):
                frame_obj = frames[i]
                if hasattr(frame_obj, "Skeleton"):
                    skel = frame_obj.Skeleton
                    # If multiple skeletons, pick the first one (usually the user)
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]

                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition is usually 20x1 struct array or similar
                        # We need to extract X, Y, Z for each joint
                        # Assuming JointsType order is fixed as per prompt

                        # Check if WorldPosition is an object or array
                        wp = skel.WorldPosition
                        if isinstance(wp, np.ndarray) and wp.size == Config.NUM_JOINTS:
                            for j in range(Config.NUM_JOINTS):
                                joint = wp[j]
                                skeleton_data[i, j, 0] = joint.X
                                skeleton_data[i, j, 1] = joint.Y
                                skeleton_data[i, j, 2] = joint.Z

            # Pose: Relative to HipCenter (Index 0)
            hip_center = skeleton_data[:, 0:1, :]
            pose = skeleton_data - hip_center
            pose = pose.reshape(num_frames, -1)  # Flatten to (T, 60)
            pose = torch.tensor(pose, dtype=torch.float32)

            # Cite solution_lesson_node_00053: Removed explicit velocity calculation

            # --- Label Processing ---
            labels = torch.zeros(num_frames, dtype=torch.long)

            # Extract labels from metadata or MAT
            # The metadata CSV has a 'labels' string, but that doesn't give start/end frames.
            # We must use the MAT file 'Labels' struct for precise frame-wise annotation.
            if hasattr(video_struct, "Labels"):
                lbls = video_struct.Labels
                if not isinstance(lbls, np.ndarray):
                    lbls = np.array([lbls]) if lbls is not None else np.array([])

                if lbls.size > 0:
                    # If single element, wrap in list
                    if lbls.size == 1 and not isinstance(lbls, np.ndarray):
                        lbls = [lbls]
                    elif lbls.size == 1:
                        lbls = [lbls.item()]

                    for l in lbls:
                        if (
                            hasattr(l, "Name")
                            and hasattr(l, "Begin")
                            and hasattr(l, "End")
                        ):
                            name = l.Name
                            if name in Config.LABEL_MAP:
                                lid = Config.LABEL_MAP[name]
                                start = max(0, int(l.Begin) - 1)  # 1-based to 0-based
                                end = min(num_frames, int(l.End))
                                labels[start:end] = lid

        except Exception as e:
            print(f"Error processing MAT {mat_path}: {e}")
            # Return dummy data to avoid crash, will likely be filtered out or cause poor performance
            # Ideally this shouldn't happen with valid data
            num_frames = 100
            pose = torch.zeros((num_frames, Config.POSE_INPUT_DIM))
            labels = torch.zeros(num_frames, dtype=torch.long)

        # 2. Process Audio
        audio_path = (
            os.path.join(Config.INPUT_DIR, row["audio_path"])
            if pd.notna(row["audio_path"])
            else None
        )
        audio_features = torch.zeros((num_frames, Config.AUDIO_INPUT_DIM))

        if audio_path and os.path.exists(audio_path):
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # Resample if needed
                if sample_rate != Config.AUDIO_SAMPLE_RATE:
                    resampler = T.Resample(sample_rate, Config.AUDIO_SAMPLE_RATE)
                    waveform = resampler(waveform)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Extract MFCC
                mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)
                mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

                # Cite solution_lesson_node_00054: Removed interpolation, relying on hop_length alignment
                audio_features = mfcc
            except Exception as e:
                print(f"Error processing Audio {audio_path}: {e}")

        # Ensure lengths match exactly
        min_len = min(pose.shape[0], audio_features.shape[0])
        # Truncate to min length to handle rounding errors from audio extraction
        pose = pose[:min_len]
        labels = labels[:min_len]
        audio_features = audio_features[:min_len]

        # If audio is shorter than video (rare with center=False/True logic but possible), pad
        if audio_features.shape[0] < pose.shape[0]:
            pad_len = pose.shape[0] - audio_features.shape[0]
            padding = torch.zeros((pad_len, Config.AUDIO_INPUT_DIM))
            audio_features = torch.cat([audio_features, padding], dim=0)

        return pose, audio_features, labels

    def _augment_data(self, pose, audio):
        """Applies random augmentation."""
        T, D_pose = pose.shape
        T_aud, D_aud = audio.shape

        # 1. Gaussian Noise (Pose only)
        if np.random.rand() < 0.5:
            noise = torch.randn_like(pose) * 0.01
            pose = pose + noise

        # 2. Channel Masking
        if np.random.rand() < 0.3:
            # Mask Pose channels
            mask_idx = np.random.choice(D_pose, size=int(D_pose * 0.1), replace=False)
            pose[:, mask_idx] = 0

            # Mask Audio channels
            mask_idx_aud = np.random.choice(D_aud, size=int(D_aud * 0.1), replace=False)
            audio[:, mask_idx_aud] = 0

        # 3. Temporal Cutout
        if np.random.rand() < 0.3:
            # Mask a contiguous chunk
            mask_len = np.random.randint(5, 15)
            if T > mask_len:
                start = np.random.randint(0, T - mask_len)
                pose[start : start + mask_len, :] = 0
                audio[start : start + mask_len, :] = 0

        return pose, audio


def collate_fn(batch):
    """
    Pads sequences to the longest in the batch.
    Returns:
        pose: (B, T_max, D_p)
        audio: (B, T_max, D_a)
        labels: (B, T_max)
        lengths: (B,)
    """
    poses, audios, labels = zip(*batch)

    lengths = torch.tensor([p.shape[0] for p in poses])

    pose_padded = pad_sequence(poses, batch_first=True, padding_value=0)
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background

    return (
        pose_padded,
        audio_padded,
        labels_padded,
        lengths,
    )


def compute_global_stats(metadata_df):
    """
    Computes global mean and std for Pose and Audio streams.
    Uses the cache if stats.npz exists, otherwise computes from scratch.
    """
    stats_path = os.path.join(Config.WORKING_DIR, "stats.npz")

    if os.path.exists(stats_path):
        print("Loading global stats from cache...")
        loaded = np.load(stats_path)
        return {k: torch.tensor(v) for k, v in loaded.items()}

    print("Computing global stats from training data...")
    dataset = GestureDataset(
        metadata_df, mode="test", stats=None, load_cached_data=True
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # Accumulators
    pose_sum = torch.zeros(Config.POSE_INPUT_DIM)
    pose_sq_sum = torch.zeros(Config.POSE_INPUT_DIM)
    audio_sum = torch.zeros(Config.AUDIO_INPUT_DIM)
    audio_sq_sum = torch.zeros(Config.AUDIO_INPUT_DIM)

    total_frames = 0

    for pose, audio, _, lengths in loader:
        # Unpad
        L = lengths[0]
        p = pose[0, :L]
        a = audio[0, :L]

        pose_sum += p.sum(dim=0)
        pose_sq_sum += (p**2).sum(dim=0)

        audio_sum += a.sum(dim=0)
        audio_sq_sum += (a**2).sum(dim=0)

        total_frames += L

    def get_stats(s, sq, N):
        mean = s / N
        var = (sq / N) - (mean**2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return mean, std

    pose_mean, pose_std = get_stats(pose_sum, pose_sq_sum, total_frames)
    audio_mean, audio_std = get_stats(audio_sum, audio_sq_sum, total_frames)

    stats = {
        "pose_mean": pose_mean,
        "pose_std": pose_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }

    # Save to cache
    np.savez(stats_path, **{k: v.numpy() for k, v in stats.items()})

    return stats


def get_dataloaders():
    """
    Creates DataLoaders for Train, Val, and Test sets.
    """
    set_seed(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Compute Stats (using only training data)
    stats = compute_global_stats(train_df)

    # Datasets
    train_ds = GestureDataset(train_df, mode="train", stats=stats)
    val_ds = GestureDataset(val_df, mode="val", stats=stats)
    test_ds = GestureDataset(test_df, mode="test", stats=stats)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Sequential for inference
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
