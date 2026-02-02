import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WINDOW_SIZE,
    STRIDE,
    AUDIO_SAMPLE_RATE,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    SKELETON_JOINTS,
    BACKGROUND_CLASS_ID,
    DEBUG_DATA_LIMIT,
)
from library.utils import set_seed


class PolymorphicParser:
    """
    Robustly parses .mat files to extract Skeleton World Positions.
    Handles variations in struct/cell array formats common in MATLAB exports.
    """

    @staticmethod
    def parse_mat(mat_path):
        try:
            # Load with squeeze_me=True to simplify arrays, struct_as_record=False to get objects
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            if "Video" not in mat:
                return None

            video = mat["Video"]
            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames
            num_frames = getattr(video, "NumFrames", 0)

            # Initialize container: T x Joints x 3
            # We assume 20 joints based on config
            skeleton_data = np.zeros((num_frames, SKELETON_JOINTS, 3), dtype=np.float32)

            # Normalize frames to a list-like structure
            if isinstance(frames, np.ndarray):
                frame_list = frames
            elif isinstance(frames, list):
                frame_list = frames
            else:
                # Single object or unknown, wrap it if iterable or treat as single
                if hasattr(frames, "__len__") and not isinstance(frames, str):
                    frame_list = frames
                else:
                    frame_list = [frames]

            # Iterate and extract
            last_valid_frame = np.zeros((SKELETON_JOINTS, 3), dtype=np.float32)

            # Determine how many frames to process
            count = min(len(frame_list), num_frames)

            for i in range(count):
                f = frame_list[i]
                has_skel = False

                if hasattr(f, "Skeleton"):
                    skel = f.Skeleton
                    # Check if Skeleton is valid (sometimes it's empty or UserIndex is 0)
                    # We check if WorldPosition exists
                    if hasattr(skel, "WorldPosition"):
                        wp = skel.WorldPosition
                        # Extract X, Y, Z
                        # wp might be a struct with X,Y,Z fields or an array
                        try:
                            if (
                                hasattr(wp, "X")
                                and hasattr(wp, "Y")
                                and hasattr(wp, "Z")
                            ):
                                # MATLAB structs often come as single values per joint or arrays
                                # We expect 20 joints.
                                # If wp.X is scalar, this is wrong. If wp.X is array of 20, good.
                                x = np.atleast_1d(wp.X)
                                y = np.atleast_1d(wp.Y)
                                z = np.atleast_1d(wp.Z)

                                if len(x) == SKELETON_JOINTS:
                                    # Stack
                                    current_pose = np.stack([x, y, z], axis=1)  # 20x3
                                    skeleton_data[i] = current_pose
                                    last_valid_frame = current_pose
                                    has_skel = True
                        except Exception:
                            pass

                if not has_skel:
                    # Fill with last valid data (simple interpolation/hold)
                    skeleton_data[i] = last_valid_frame

            # If num_frames > len(frame_list), pad the rest
            for i in range(count, num_frames):
                skeleton_data[i] = last_valid_frame

            return skeleton_data

        except Exception as e:
            # print(f"Error parsing {mat_path}: {e}")
            return None


class AudioProcessor:
    """
    Handles loading and MFCC extraction for audio, ensuring alignment with video frames.
    """

    def __init__(self):
        self.mfcc_transform = T.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "n_mels": 64,
                "hop_length": HOP_LENGTH,
                "mel_scale": "htk",
            },
        )

    def process(self, audio_path, target_num_frames):
        if not os.path.exists(audio_path):
            return np.zeros((target_num_frames, N_MFCC), dtype=np.float32)

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if necessary
            if sr != AUDIO_SAMPLE_RATE:
                resampler = T.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Compute MFCC: (Channel, n_mfcc, time)
            mfcc = self.mfcc_transform(waveform)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = mfcc.mean(dim=0, keepdim=True)

            # Shape: (1, n_mfcc, time) -> (1, time, n_mfcc)
            mfcc = mfcc.transpose(1, 2)

            # Align with video frames using interpolation
            # Input to interpolate needs to be (Batch, Channels, Time)
            # Here 'Channels' is n_mfcc.
            # Current: (1, time, n_mfcc) -> permute to (1, n_mfcc, time)
            mfcc = mfcc.permute(0, 2, 1)

            if target_num_frames > 0:
                mfcc = F.interpolate(
                    mfcc, size=target_num_frames, mode="linear", align_corners=False
                )

            # Back to (time, n_mfcc) -> (target_num_frames, n_mfcc)
            mfcc = mfcc.squeeze(0).transpose(0, 1)

            return mfcc.numpy()

        except Exception:
            return np.zeros((target_num_frames, N_MFCC), dtype=np.float32)


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Processes the raw data defined in metadata_path.
    Extracts Skeleton, Audio, and Labels.
    Caches the result in .npz format.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary from npz
            # Keys are like 'SampleID_type'
            samples = {}
            keys = sorted(data.files)
            # Group keys by sample ID
            # Assuming format: {sample_id}_skel, {sample_id}_audio, {sample_id}_label
            # We can just iterate metadata to know what keys to look for
            df = pd.read_csv(metadata_path)
            if DEBUG_DATA_LIMIT:
                df = df.head(DEBUG_DATA_LIMIT)

            loaded_data = []
            for _, row in df.iterrows():
                sid = row["sample_id"]
                if f"{sid}_skel" in data:
                    samples = {
                        "sample_id": sid,
                        "skeleton": data[f"{sid}_skel"],
                        "audio": data[f"{sid}_audio"],
                        "labels": data[f"{sid}_label"],
                    }
                    loaded_data.append(samples)
            return loaded_data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Processing from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    if DEBUG_DATA_LIMIT:
        df = df.head(DEBUG_DATA_LIMIT)

    # Parse JSON labels
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    parser = PolymorphicParser()
    audio_proc = AudioProcessor()

    processed_samples = []
    export_dict = {}

    for idx, row in df.iterrows():
        sid = row["sample_id"]
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # 1. Parse Skeleton
        skeleton = parser.parse_mat(mat_path)
        if skeleton is None:
            # Fallback if parsing fails totally (should be rare with robust parser)
            # Assume some default length or skip? Better to skip to avoid crashing training
            continue

        num_frames = skeleton.shape[0]

        # 2. Process Audio
        audio = audio_proc.process(audio_path, num_frames)

        # 3. Process Labels
        # Initialize with Background (0)
        labels = np.full(num_frames, BACKGROUND_CLASS_ID, dtype=np.int64)
        for annot in row["parsed_labels"]:
            gid = annot["id"]
            start = max(0, annot["begin"] - 1)  # 1-based to 0-based
            end = min(num_frames, annot["end"])
            if start < end:
                labels[start:end] = gid

        sample_data = {
            "sample_id": sid,
            "skeleton": skeleton,
            "audio": audio,
            "labels": labels,
        }
        processed_samples.append(sample_data)

        # Add to export dict
        export_dict[f"{sid}_skel"] = skeleton
        export_dict[f"{sid}_audio"] = audio
        export_dict[f"{sid}_label"] = labels

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, **export_dict)
    print(f"Saved processed data to {cache_path}")

    return processed_samples


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Multi-modal Gesture Recognition.
    Implements Kinematically Consistent Augmentation and Sliding Windows.
    """

    def __init__(self, samples, is_train=True, window_size=WINDOW_SIZE, stride=STRIDE):
        self.is_train = is_train
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.data_source = samples  # List of dicts

        # Pre-calculate windows
        for i, sample in enumerate(samples):
            num_frames = sample["skeleton"].shape[0]
            if num_frames < window_size:
                # Pad short sequences? Or just take one window with padding?
                # For simplicity, we'll just take one window starting at 0 and handle padding in __getitem__
                self.windows.append((i, 0))
            else:
                # Sliding window
                for start in range(0, num_frames - window_size + 1, stride):
                    self.windows.append((i, start))

                # Ensure the last frames are covered if not perfectly divisible
                if (num_frames - window_size) % stride != 0:
                    self.windows.append((i, num_frames - window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.windows[idx]
        sample = self.data_source[sample_idx]

        # Extract raw data
        raw_skel = sample["skeleton"]  # T x 20 x 3
        raw_audio = sample["audio"]  # T x n_mfcc
        raw_labels = sample["labels"]  # T

        seq_len = raw_skel.shape[0]

        # Handle padding for short sequences
        if seq_len < self.window_size:
            # Pad with zeros
            pad_len = self.window_size - seq_len

            skel_window = np.pad(
                raw_skel, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
            )
            audio_window = np.pad(raw_audio, ((0, pad_len), (0, 0)), mode="constant")
            label_window = np.pad(
                raw_labels,
                (0, pad_len),
                mode="constant",
                constant_values=BACKGROUND_CLASS_ID,
            )
        else:
            end_frame = start_frame + self.window_size
            skel_window = raw_skel[start_frame:end_frame].copy()
            audio_window = raw_audio[start_frame:end_frame].copy()
            label_window = raw_labels[start_frame:end_frame].copy()

        # Kinematically Consistent Augmentation
        if self.is_train:
            # 1. Random Rotation around Y-axis
            theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            # Apply rotation: (T, J, 3) dot (3, 3) -> (T, J, 3)
            skel_window = np.dot(skel_window, R.T)

            # 2. Random Scaling
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

            # Add slight Gaussian noise to positions
            noise = np.random.normal(0, 0.001, skel_window.shape).astype(np.float32)
            skel_window += noise

        # Derive Velocity and Acceleration from (Augmented) Positions
        # skel_window is (T, 20, 3)
        # Velocity: diff along time
        vel = np.zeros_like(skel_window)
        vel[1:] = skel_window[1:] - skel_window[:-1]

        # Acceleration: diff of velocity
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Flatten skeleton features: (T, 20*3)
        pos_flat = skel_window.reshape(self.window_size, -1)
        vel_flat = vel.reshape(self.window_size, -1)
        acc_flat = acc.reshape(self.window_size, -1)

        # Concatenate all features: Pos + Vel + Acc + Audio
        # Shape: (T, 60 + 60 + 60 + 13) = (T, 193)
        features = np.concatenate([pos_flat, vel_flat, acc_flat, audio_window], axis=1)

        return {
            "features": torch.from_numpy(features).float(),
            "labels": torch.from_numpy(label_window).long(),
        }


def get_dataloaders(batch_size=32, load_cached_data=True):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    # Load processed data
    train_samples = process_dataset(
        TRAIN_METADATA_PATH, "dataset_train", load_cached_data
    )
    val_samples = process_dataset(VAL_METADATA_PATH, "dataset_val", load_cached_data)
    test_samples = process_dataset(TEST_METADATA_PATH, "dataset_test", load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(
        train_samples, is_train=True, window_size=WINDOW_SIZE, stride=STRIDE
    )
    # Validation uses sliding windows too for loss calculation, but no augmentation
    val_dataset = GestureDataset(
        val_samples, is_train=False, window_size=WINDOW_SIZE, stride=STRIDE
    )
    # Test dataset for inference
    test_dataset = GestureDataset(
        test_samples, is_train=False, window_size=WINDOW_SIZE, stride=STRIDE // 2
    )  # Overlap for inference

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_samples
