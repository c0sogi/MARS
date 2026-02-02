import os
import numpy as np
import torch
import pandas as pd
import scipy.io
import scipy.ndimage
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    UPPER_BODY_JOINTS,
    GESTURE_MAP,
    SCALE_FACTOR,
    NUM_MFCC,
    NOISE_STD,
    TEMPORAL_FILTER_WIDTH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SEED,
)
from library.utils import set_seed

# Ensure reproducible behavior
set_seed(SEED)


def extract_skeleton_features(mat_path):
    """
    Parses .mat file to extract and normalize skeleton joints.
    Returns:
        positions (np.ndarray): (T, 12, 3) normalized positions in meters.
        num_frames (int): Number of frames.
        labels_list (list): List of label objects (Name, Begin, End) if available.
    """
    try:
        # Load mat file
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, 0, []

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames_data = getattr(video, "Frames", [])
        labels_raw = getattr(video, "Labels", [])

        # Handle Labels structure (can be single object, array, or empty)
        labels_list = []
        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0 and labels_raw.size > 0:  # 0-d array
                labels_list.append(labels_raw.item())
            else:
                for l in labels_raw:
                    labels_list.append(l)
        elif hasattr(labels_raw, "Name"):  # Single object
            labels_list.append(labels_raw)

        # Extract Skeleton Data
        # frames_data is usually an array of structures.
        # We need to extract WorldPosition for specific joints.
        # Structure: Frames[t].Skeleton.WorldPosition (or similar depending on nesting)

        # Pre-allocate
        # 20 joints total in raw data
        full_skeleton = np.zeros((num_frames, 20, 3), dtype=np.float32)

        # Check if frames_data is valid
        if isinstance(frames_data, np.ndarray) and len(frames_data) == num_frames:
            for t in range(num_frames):
                skel = frames_data[t].Skeleton
                # skel might be an array of skeletons (users), usually we take the first tracked one
                # The description says "User Index" map exists, but usually for these datasets
                # the Skeleton field contains the tracked user.
                # If skel is an array, we take the first one.
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    curr_skel = skel[0]
                else:
                    curr_skel = skel

                # Extract WorldPosition
                # WorldPosition is often a struct with x, y, z or an array
                # Based on description: "WorldPosition... formed by 20x4 matrix"?
                # Actually description says "WorldPosition... X, Y, Z".
                # Let's assume standard Kinect format where we have joints.

                # In many MATLAB exports of this dataset, WorldPosition is a field inside the joint struct
                # or Skeleton has a field WorldPosition which is 20x3.
                # Let's try to inspect the object structure dynamically or assume common format.
                # Given the description: "Skeleton Frame... contains JointsType, WorldPosition..."
                # It implies Skeleton is an array of joints.

                if hasattr(curr_skel, "WorldPosition"):
                    # If Skeleton is a struct containing arrays
                    wp = curr_skel.WorldPosition  # Expected 20x3 or similar
                    if isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                        full_skeleton[t] = wp
                    elif isinstance(wp, np.ndarray) and wp.shape == (3, 20):
                        full_skeleton[t] = wp.T
                    else:
                        # Fallback: maybe Skeleton is an array of Joint objects
                        pass
                elif isinstance(curr_skel, np.ndarray):
                    # Skeleton might be array of 20 joint objects
                    for j_idx in range(min(len(curr_skel), 20)):
                        joint = curr_skel[j_idx]
                        if hasattr(joint, "WorldPosition"):
                            pos = joint.WorldPosition
                            full_skeleton[t, j_idx, 0] = pos.X
                            full_skeleton[t, j_idx, 1] = pos.Y
                            full_skeleton[t, j_idx, 2] = pos.Z

        # Select Upper Body Joints
        # UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        upper_body = full_skeleton[:, UPPER_BODY_JOINTS, :]  # (T, 12, 3)

        # Normalization
        # 1. Scale mm to m
        upper_body = upper_body * SCALE_FACTOR

        # 2. Center relative to HipCenter (Index 0 in our selected subset)
        # HipCenter is index 0 in UPPER_BODY_JOINTS (dataset index 0)
        hip_center = upper_body[:, 0:1, :]  # (T, 1, 3)
        upper_body = upper_body - hip_center

        return upper_body, num_frames, labels_list

    except Exception as e:
        # print(f"Error processing {mat_path}: {e}")
        return None, 0, []


def extract_audio_features(audio_path, target_frames):
    """
    Extracts MFCC features and aligns them to video frames.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        # We want to align with video frames.
        # Video FPS is ~10-20. Audio is 16kHz.
        # We can compute MFCCs and then interpolate.

        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

        # Interpolate to match target_frames
        if mfcc.shape[0] != target_frames:
            mfcc = mfcc.unsqueeze(0).transpose(1, 2)  # (1, n_mfcc, time)
            mfcc = torch.nn.functional.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )
            mfcc = mfcc.transpose(1, 2).squeeze(0)  # (target_frames, n_mfcc)

        return mfcc.numpy()

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_frames, NUM_MFCC), dtype=np.float32)


def process_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Loads raw data, processes it, and caches it.
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["samples"].tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load metadata
    df = pd.read_csv(metadata_path)
    samples = []

    print(f"Processing {len(df)} samples for {cache_name}...")

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # Process Skeleton
        positions, num_frames, labels_obj_list = extract_skeleton_features(mat_path)

        if positions is None or num_frames == 0:
            continue

        # Process Audio
        audio_feats = extract_audio_features(audio_path, num_frames)

        # Process Labels
        # Frame-wise class labels (0=Background)
        frame_labels = np.zeros(num_frames, dtype=np.int64)
        # Boundary labels (1 at transition)
        boundary_labels = np.zeros(num_frames, dtype=np.float32)

        # Fill labels if available (Train/Val)
        if hasattr(row, "labels") and labels_obj_list:
            for obj in labels_obj_list:
                try:
                    name = obj.Name
                    start = int(obj.Begin) - 1  # 1-based to 0-based
                    end = int(obj.End) - 1

                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        # Clamp indices
                        start = max(0, start)
                        end = min(num_frames - 1, end)

                        if start <= end:
                            frame_labels[start : end + 1] = gid
                            boundary_labels[start] = 1.0
                            boundary_labels[end] = 1.0
                except AttributeError:
                    continue

        samples.append(
            {
                "sample_id": sample_id,
                "positions": positions.astype(np.float32),  # (T, 12, 3)
                "audio": audio_feats.astype(np.float32),  # (T, 13)
                "frame_labels": frame_labels,  # (T,)
                "boundary_labels": boundary_labels,  # (T,)
            }
        )

    # Save to cache
    print(f"Saving {len(samples)} samples to {cache_path}")
    np.savez_compressed(cache_path, samples=np.array(samples, dtype=object))

    return samples


class GestureDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        positions = sample["positions"].copy()  # (T, 12, 3)
        audio = sample["audio"].copy()  # (T, 13)
        frame_labels = sample["frame_labels"]
        boundary_labels = sample["boundary_labels"]

        T = positions.shape[0]

        # Augmentation (Train only)
        if self.is_train:
            # Physically consistent augmentation
            # 1. Generate Gaussian Noise
            noise = np.random.normal(0, NOISE_STD, positions.shape)

            # 2. Apply Temporal Low-Pass Filter
            # Apply along time axis (axis 0)
            smooth_noise = scipy.ndimage.gaussian_filter1d(
                noise, sigma=TEMPORAL_FILTER_WIDTH, axis=0
            )

            # 3. Add to positions
            positions = positions + smooth_noise

        # Feature Engineering: Velocity
        # V_t = P_t - P_{t-1}
        # Pad first frame with 0 velocity
        velocity = np.zeros_like(positions)
        velocity[1:] = positions[1:] - positions[:-1]

        # Flatten Spatial Dimensions
        # Positions: (T, 12, 3) -> (T, 36)
        # Velocity: (T, 12, 3) -> (T, 36)
        pos_flat = positions.reshape(T, -1)
        vel_flat = velocity.reshape(T, -1)

        # Concatenate Features
        # [Pos, Vel, Audio] -> (T, 36+36+13) = (T, 85)
        features = np.concatenate([pos_flat, vel_flat, audio], axis=1)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "cls_labels": torch.tensor(frame_labels, dtype=torch.long),
            "bnd_labels": torch.tensor(boundary_labels, dtype=torch.float32),
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Collates a batch of variable length sequences.
    """
    features_list = [item["features"] for item in batch]
    cls_labels_list = [item["cls_labels"] for item in batch]
    bnd_labels_list = [item["bnd_labels"] for item in batch]
    ids = [item["sample_id"] for item in batch]

    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> (B, T, D)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0)
    cls_labels_padded = pad_sequence(cls_labels_list, batch_first=True, padding_value=0)
    bnd_labels_padded = pad_sequence(bnd_labels_list, batch_first=True, padding_value=0)

    # Create Mask (B, T)
    # 1 for valid, 0 for padding
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()

    return {
        "features": features_padded,  # (B, T, D)
        "cls_labels": cls_labels_padded,  # (B, T)
        "bnd_labels": bnd_labels_padded,  # (B, T)
        "mask": mask,  # (B, T)
        "lengths": lengths,  # (B,)
        "sample_ids": ids,
    }


def get_dataloaders(batch_size=8, load_cached_data=True):
    """
    Factory function to get train, val, and test dataloaders.
    """
    # Process/Load Data
    train_samples = process_dataset(TRAIN_METADATA_PATH, "train_data", load_cached_data)
    val_samples = process_dataset(VAL_METADATA_PATH, "val_data", load_cached_data)
    test_samples = process_dataset(TEST_METADATA_PATH, "test_data", load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(train_samples, is_train=True)
    val_dataset = GestureDataset(val_samples, is_train=False)
    test_dataset = GestureDataset(test_samples, is_train=False)

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
