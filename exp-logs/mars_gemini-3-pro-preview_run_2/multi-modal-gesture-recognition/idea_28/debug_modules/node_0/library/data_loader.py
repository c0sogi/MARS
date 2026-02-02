import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    GESTURE_MAP,
    SKELETON_JOINTS,
    NUM_JOINTS,
    JOINT_DIM,
    SKELETON_SCALE_FACTOR,
    AUDIO_SAMPLE_RATE,
    NUM_MFCC,
    SEED,
)
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(SEED)


def physically_consistent_augmentation(skeleton_pos, sigma=0.005, temporal_sigma=1.0):
    """
    Applies temporally correlated noise to joint positions.

    Args:
        skeleton_pos (np.ndarray): Shape (T, J, 3).
        sigma (float): Standard deviation of the Gaussian noise (in meters).
        temporal_sigma (float): Sigma for the temporal Gaussian filter.

    Returns:
        np.ndarray: Augmented skeleton positions.
    """
    # Generate independent Gaussian noise
    noise = np.random.normal(loc=0.0, scale=sigma, size=skeleton_pos.shape)

    # Apply temporal smoothing to the noise to make it physically plausible (sensor drift/jitter)
    # We apply the filter along axis 0 (time)
    smooth_noise = scipy.ndimage.gaussian_filter1d(noise, sigma=temporal_sigma, axis=0)

    return skeleton_pos + smooth_noise


def process_sample(sample_info):
    """
    Parses raw data files to extract skeleton, audio, and label information.

    Args:
        sample_info (dict/Series): Row from the metadata dataframe.

    Returns:
        tuple: (skeleton_pos, audio_mfcc, frame_labels, boundaries) or None if failed.
    """
    # 1. Load Skeleton Data
    mat_path = os.path.join(INPUT_DIR, sample_info["data_path"])
    try:
        # Load mat file with robust struct handling
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        frames = getattr(video, "Frames", [])

        # Handle case where Frames is a single object or list/array
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        # Initialize skeleton array: (T, J, 3)
        skeleton_pos = np.zeros((num_frames, NUM_JOINTS, JOINT_DIM), dtype=np.float32)

        for t, frame in enumerate(frames):
            if t >= num_frames:
                break

            skel = getattr(frame, "Skeleton", None)
            if skel is None:
                continue

            # Robustly extract joint positions
            # Assuming skel is indexable (array of joints) or we need to map indices
            # Based on standard Kinect structure in .mat files for this dataset type:
            try:
                for j_idx, joint_idx in enumerate(SKELETON_JOINTS):
                    # Check if skel is iterable (array of joints)
                    if isinstance(skel, (np.ndarray, list)):
                        if joint_idx < len(skel):
                            joint = skel[joint_idx]
                            wp = getattr(joint, "WorldPosition", None)
                            if wp:
                                skeleton_pos[t, j_idx, 0] = wp.X
                                skeleton_pos[t, j_idx, 1] = wp.Y
                                skeleton_pos[t, j_idx, 2] = wp.Z
            except Exception:
                pass

    except Exception as e:
        # print(f"Error processing skeleton for {sample_info.get('sample_id')}: {e}")
        return None

    # Preprocessing: Center and Scale
    # 1. Center around HipCenter (Index 0 in SKELETON_JOINTS if 0 is HipCenter)
    # We assume SKELETON_JOINTS[0] corresponds to the root joint used for centering
    hip_center = skeleton_pos[:, 0:1, :]  # (T, 1, 3)
    skeleton_pos = skeleton_pos - hip_center

    # 2. Scale units (mm -> meters)
    skeleton_pos = skeleton_pos * SKELETON_SCALE_FACTOR

    # 2. Load Audio Data
    audio_path = os.path.join(INPUT_DIR, sample_info["audio_path"])
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # Average channels -> (n_mfcc, time)

        # Align Audio to Video Frames using interpolation
        mfcc = mfcc.unsqueeze(0)  # (1, F, T_audio)
        mfcc_aligned = F.interpolate(
            mfcc, size=num_frames, mode="linear", align_corners=False
        )
        mfcc_aligned = mfcc_aligned.squeeze(0).permute(1, 0).numpy()  # (T_video, F)

    except Exception:
        # Fallback: Zero audio features
        mfcc_aligned = np.zeros((num_frames, NUM_MFCC), dtype=np.float32)

    # 3. Process Labels
    frame_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (Background)
    boundaries = np.zeros(num_frames, dtype=np.float32)

    # Only process labels if they exist (Train/Val)
    if "labels" in sample_info:
        try:
            video = mat["Video"]
            labels_raw = getattr(video, "Labels", [])

            def process_lbl(obj):
                try:
                    name = obj.Name
                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        # Convert 1-based indexing to 0-based
                        start = int(obj.Begin) - 1
                        end = int(obj.End)

                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames, end)

                        if start < end:
                            frame_labels[start:end] = gid
                            # Mark boundary at the start frame
                            if start < num_frames:
                                boundaries[start] = 1.0
                except AttributeError:
                    pass

            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_lbl(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_lbl(l)
            elif isinstance(labels_raw, list):
                for l in labels_raw:
                    process_lbl(l)
            else:
                process_lbl(labels_raw)
        except Exception:
            pass

    return (
        skeleton_pos.astype(np.float32),
        mfcc_aligned.astype(np.float32),
        frame_labels,
        boundaries,
    )


def prepare_dataset(metadata_path, cache_name, load_cached_data=True):
    """
    Loads dataset from metadata, processing samples and caching the result.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        cache_name (str): Identifier for the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (positions, audios, labels, boundaries, ids) as numpy object arrays.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["positions"],
                data["audios"],
                data["labels"],
                data["boundaries"],
                data["ids"],
            )
        except Exception:
            pass  # Fallback to reprocessing if load fails

    # print(f"Processing data from {metadata_path}")
    df = pd.read_csv(metadata_path)

    positions_list = []
    audios_list = []
    labels_list = []
    boundaries_list = []
    ids_list = []

    for _, row in df.iterrows():
        res = process_sample(row)
        if res is None:
            continue

        pos, aud, lbl, bnd = res
        positions_list.append(pos)
        audios_list.append(aud)
        labels_list.append(lbl)
        boundaries_list.append(bnd)
        ids_list.append(row["sample_id"])

    # Convert to object arrays to handle variable lengths
    positions_arr = np.array(positions_list, dtype=object)
    audios_arr = np.array(audios_list, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)
    boundaries_arr = np.array(boundaries_list, dtype=object)
    ids_arr = np.array(ids_list, dtype=object)

    # Save to cache
    np.savez_compressed(
        cache_path,
        positions=positions_arr,
        audios=audios_arr,
        labels=labels_arr,
        boundaries=boundaries_arr,
        ids=ids_arr,
    )

    return positions_arr, audios_arr, labels_arr, boundaries_arr, ids_arr


class GestureDataset(Dataset):
    """
    PyTorch Dataset that constructs features on-the-fly, allowing for dynamic augmentation.
    """

    def __init__(self, positions, audios, labels, boundaries, augment=False):
        self.positions = positions
        self.audios = audios
        self.labels = labels
        self.boundaries = boundaries
        self.augment = augment

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx]  # (T, J, 3)
        aud = self.audios[idx]  # (T, F)
        lbl = self.labels[idx]  # (T,)
        bnd = self.boundaries[idx]  # (T,)

        # Apply Augmentation if enabled
        if self.augment:
            pos = physically_consistent_augmentation(pos)

        # Compute Velocity (T, J, 3)
        # Velocity is difference between frames. Pad first frame with 0.
        velocity = np.zeros_like(pos)
        velocity[1:] = pos[1:] - pos[:-1]

        # Flatten Skeleton Features: (T, J*3)
        T = pos.shape[0]
        pos_flat = pos.reshape(T, -1)
        vel_flat = velocity.reshape(T, -1)

        # Concatenate all features: [Position, Velocity, Audio]
        # Shape: (T, D)
        features = np.concatenate([pos_flat, vel_flat, aud], axis=1)

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(lbl.astype(np.int64)),
            torch.from_numpy(bnd.astype(np.float32)),
        )


def collate_fn(batch):
    """
    Pads sequences to the maximum length in the batch.

    Returns:
        padded_features: (B, D, T_max) - Permuted for Conv1d
        padded_labels: (B, T_max)
        padded_boundaries: (B, T_max)
        mask: (B, T_max)
    """
    features, labels, boundaries = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    batch_size = len(features)
    feature_dim = features[0].shape[1]

    # Initialize padded tensors
    feat_pad = torch.zeros(batch_size, feature_dim, max_len, dtype=torch.float32)
    lbl_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    bnd_pad = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (f, l, b) in enumerate(zip(features, labels, boundaries)):
        end = lengths[i]
        # Transpose features to (D, T) for PyTorch Conv1d compatibility
        feat_pad[i, :, :end] = f.permute(1, 0)
        lbl_pad[i, :end] = l
        bnd_pad[i, :end] = b
        mask[i, :end] = 1

    return feat_pad, lbl_pad, bnd_pad, mask
