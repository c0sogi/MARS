import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.io
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    CACHE_FILE_PREFIX,
    UPPER_BODY_JOINTS,
    HIP_CENTER_INDEX,
    SKELETON_EDGES,
    SCALE_FACTOR,
    AUDIO_MFCC_DIM,
    MAX_SEQ_LEN,
    GESTURE_MAP,
    GAUSSIAN_SIGMA,
    CACHE_DATA,
    SEED,
)
from library.utils import parse_label_string, set_seed

# Ensure reproducible behavior
set_seed(SEED)


def compute_gaussian_targets(num_frames, transitions, sigma=GAUSSIAN_SIGMA):
    """
    Generates a 1D sequence of Gaussian bumps centered at transition frames.
    """
    targets = np.zeros(num_frames, dtype=np.float32)
    if not transitions:
        return targets

    x = np.arange(num_frames)
    for t in transitions:
        # Gaussian function: exp(-0.5 * ((x - mu) / sigma)^2)
        # We use max to combine overlapping Gaussians (though rare in this data)
        g = np.exp(-0.5 * ((x - t) / sigma) ** 2)
        targets = np.maximum(targets, g)
    return targets


def load_sample_data(row):
    """
    Loads and preprocesses a single sample from raw files.
    Returns a dictionary containing:
        - pos: Normalized joint positions (T, 12, 3)
        - audio: Aligned MFCC features (T, 13)
        - cls_targets: Frame-wise class labels (T,)
        - bnd_targets: Frame-wise boundary scores (T,)
    """
    sample_id = row["sample_id"]
    data_path = os.path.join(INPUT_DIR, row["data_path"])
    audio_path = os.path.join(INPUT_DIR, row["audio_path"])

    # 1. Load MAT file
    try:
        mat = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        num_frames = video.NumFrames
        frames = video.Frames

        # Handle Labels
        # Labels might be missing in test data
        labels_raw = getattr(video, "Labels", [])
        gesture_instances = []

        # Helper to parse label struct
        def parse_label_obj(obj):
            try:
                name = obj.Name
                start = obj.Begin
                end = obj.End
                if name in GESTURE_MAP:
                    return (GESTURE_MAP[name], start, end)
            except AttributeError:
                pass
            return None

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.ndim == 0:
                g = parse_label_obj(labels_raw.item())
                if g:
                    gesture_instances.append(g)
            else:
                for l in labels_raw:
                    g = parse_label_obj(l)
                    if g:
                        gesture_instances.append(g)
        elif hasattr(labels_raw, "Name"):
            g = parse_label_obj(labels_raw)
            if g:
                gesture_instances.append(g)

    except Exception as e:
        print(f"Error loading MAT for {sample_id}: {e}")
        return None

    # 2. Extract Skeleton Data (Positions)
    # We need to extract UPPER_BODY_JOINTS (indices 0-11)
    # WorldPosition is in mm.
    # Structure: Frames is an array of structs. Each has Skeleton -> WorldPosition

    # Pre-allocate
    pos_data = np.zeros((num_frames, len(UPPER_BODY_JOINTS), 3), dtype=np.float32)

    # If Frames is a single object (1 frame video), handle it, otherwise iterate
    if num_frames == 1:
        frame_list = [frames]
    else:
        frame_list = frames

    try:
        for t, frame in enumerate(frame_list):
            if t >= num_frames:
                break

            # Check if Skeleton exists and has data
            if hasattr(frame, "Skeleton"):
                skel = frame.Skeleton
                # skel is an array of joint structs? Or struct of arrays?
                # Description: "An array of Skeleton structures... JointsType, WorldPosition..."
                # Usually in these datasets, skel is an array of 20 joints.

                if isinstance(skel, np.ndarray):
                    # Iterate over selected joints
                    for i, joint_idx in enumerate(UPPER_BODY_JOINTS):
                        if joint_idx < len(skel):
                            joint = skel[joint_idx]
                            if hasattr(joint, "WorldPosition"):
                                wp = joint.WorldPosition
                                pos_data[t, i, 0] = wp.X
                                pos_data[t, i, 1] = wp.Y
                                pos_data[t, i, 2] = wp.Z
    except Exception as e:
        print(f"Error parsing skeleton for {sample_id}: {e}")
        # Return zeros if failed

    # 3. Normalize Skeleton
    # Center around HipCenter (Index 0 in our subset)
    # Scale to meters
    hip_center = pos_data[:, HIP_CENTER_INDEX : HIP_CENTER_INDEX + 1, :]  # (T, 1, 3)
    pos_data = (pos_data - hip_center) * SCALE_FACTOR

    # 4. Process Audio (MFCC)
    audio_features = np.zeros((num_frames, AUDIO_MFCC_DIM), dtype=np.float32)
    if os.path.exists(audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Compute MFCC
            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=AUDIO_MFCC_DIM,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = transform(waveform)  # (1, n_mfcc, time)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)

            # Align to video frames
            if mfcc.shape[0] > 0:
                mfcc_tensor = (
                    torch.from_numpy(mfcc).unsqueeze(0).transpose(1, 2)
                )  # (1, n_mfcc, time)
                # Interpolate expects (N, C, L)
                aligned = F.interpolate(
                    mfcc_tensor, size=num_frames, mode="linear", align_corners=False
                )
                audio_features = aligned.squeeze(0).transpose(0, 1).numpy()
        except Exception as e:
            # print(f"Audio error for {sample_id}: {e}")
            pass

    # 5. Construct Targets
    cls_targets = np.zeros(num_frames, dtype=np.int64)  # 0 is background
    transitions = []

    for gid, start, end in gesture_instances:
        # Matlab 1-based indexing -> Python 0-based
        s = max(0, start - 1)
        e = min(num_frames, end)
        if s < e:
            cls_targets[s:e] = gid
            transitions.append(s)
            transitions.append(e)

    bnd_targets = compute_gaussian_targets(num_frames, transitions)

    return {
        "pos": pos_data,
        "audio": audio_features,
        "cls_targets": cls_targets,
        "bnd_targets": bnd_targets,
        "sample_id": sample_id,
    }


def get_dataset_cache(metadata_df, cache_name):
    """
    Manages caching of the processed dataset.
    """
    cache_path = os.path.join(WORKING_DIR, f"{CACHE_FILE_PREFIX}_{cache_name}.npz")

    if CACHE_DATA and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct list of dicts
            data = []
            # We stored it as a single object array or individual arrays?
            # Let's assume we store 'ids', 'pos', 'audio', 'cls', 'bnd' arrays if lengths were equal,
            # but lengths are variable. So we likely stored an object array.
            if "data" in loaded:
                return loaded["data"].tolist()
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    print(f"Processing {len(metadata_df)} samples for {cache_name}...")
    data = []
    for _, row in metadata_df.iterrows():
        sample = load_sample_data(row)
        if sample is not None:
            data.append(sample)

    if CACHE_DATA:
        print(f"Saving cache to {cache_path}...")
        np.savez_compressed(cache_path, data=np.array(data, dtype=object))

    return data


class GestureDataset(Dataset):
    def __init__(self, metadata_path, is_train=True, augment=False, subset_size=None):
        """
        Args:
            metadata_path: Path to the CSV metadata file.
            is_train: Boolean, indicates if this is training data (has labels).
            augment: Boolean, whether to apply augmentation.
            subset_size: Int, for debugging, limit dataset size.
        """
        self.is_train = is_train
        self.augment = augment

        # Load Metadata
        df = pd.read_csv(metadata_path)
        if subset_size is not None:
            df = df.head(subset_size)

        # Determine cache name based on file name (train/val/test)
        cache_name = os.path.basename(metadata_path).replace(".csv", "")
        if subset_size:
            cache_name += f"_sub{subset_size}"

        # Load Data (Cached or Computed)
        self.data = get_dataset_cache(df, cache_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Unpack
        pos = torch.from_numpy(sample["pos"])  # (T, 12, 3)
        audio = torch.from_numpy(sample["audio"])  # (T, 13)
        cls_target = torch.from_numpy(sample["cls_targets"])  # (T,)
        bnd_target = torch.from_numpy(sample["bnd_targets"])  # (T,)

        T = pos.shape[0]

        # Augmentation (Physically Consistent)
        if self.augment:
            # 1. Generate Gaussian Noise
            noise = torch.randn_like(pos) * 0.01  # 1cm jitter

            # 2. Temporal Low-Pass Filter (Smoothing)
            # Reshape for Conv1d: (Batch, Channels, Time) -> (1, 36, T)
            noise_flat = noise.view(T, -1).transpose(0, 1).unsqueeze(0)

            # Simple average kernel size 5
            # For depthwise convolution (groups=channels), weight shape must be (channels, 1, kernel_size)
            channels = noise_flat.shape[1]
            kernel = torch.ones(channels, 1, 5) / 5.0
            kernel = kernel.to(pos.dtype)
            # Apply to each channel independently
            smoothed_noise = F.conv1d(noise_flat, kernel, padding=2, groups=channels)
            smoothed_noise = smoothed_noise.squeeze(0).transpose(0, 1).view(T, 12, 3)

            # 3. Add to Positions
            pos = pos + smoothed_noise

        # Feature Engineering

        # 1. Velocity: P_t - P_{t-1}
        # Pad first frame with 0
        vel = torch.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]

        # 2. Bone Vectors
        # SKELETON_EDGES is list of (p1, p2) indices
        # We want vector p2 - p1
        bones_list = []
        for p1, p2 in SKELETON_EDGES:
            # pos is (T, 12, 3). p1, p2 are indices 0-11
            bone_vec = pos[:, p2, :] - pos[:, p1, :]  # (T, 3)
            bones_list.append(bone_vec)
        bones = torch.stack(bones_list, dim=1)  # (T, NumBones, 3)

        # Flatten and Concatenate
        pos_flat = pos.view(T, -1)  # (T, 36)
        vel_flat = vel.view(T, -1)  # (T, 36)
        bones_flat = bones.view(T, -1)  # (T, 33)

        # Final Feature Vector
        features = torch.cat([pos_flat, vel_flat, bones_flat, audio], dim=1)  # (T, 118)

        return {
            "features": features,
            "cls_target": cls_target,
            "bnd_target": bnd_target,
            "length": T,
            "sample_id": sample["sample_id"],
        }


def collate_fn(batch):
    """
    Pads sequences to the maximum length in the batch.
    """
    # Sort by length (descending) for pack_padded_sequence if needed (optional)
    batch.sort(key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch])
    max_len = lengths.max().item()
    # Clamp max_len to MAX_SEQ_LEN to avoid OOM
    max_len = min(max_len, MAX_SEQ_LEN)

    feature_dim = batch[0]["features"].shape[1]
    batch_size = len(batch)

    # Initialize padded tensors
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    padded_cls = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_bnd = torch.zeros(batch_size, max_len, dtype=torch.float)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    sample_ids = []

    for i, item in enumerate(batch):
        l = min(item["length"], max_len)
        padded_features[i, :l, :] = item["features"][:l, :]
        padded_cls[i, :l] = item["cls_target"][:l]
        padded_bnd[i, :l] = item["bnd_target"][:l]
        mask[i, :l] = 1
        sample_ids.append(item["sample_id"])

    return padded_features, padded_cls, padded_bnd, lengths, mask, sample_ids
