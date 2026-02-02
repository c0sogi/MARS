import os
import json
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from library import config

# ==========================================
# Constants and Mappings
# ==========================================
JOINT_MAP = {
    "HipCenter": 0,
    "Spine": 1,
    "ShoulderCenter": 2,
    "Head": 3,
    "ShoulderLeft": 4,
    "ElbowLeft": 5,
    "WristLeft": 6,
    "HandLeft": 7,
    "ShoulderRight": 8,
    "ElbowRight": 9,
    "WristRight": 10,
    "HandRight": 11,
    "HipLeft": 12,
    "KneeLeft": 13,
    "AnkleLeft": 14,
    "FootLeft": 15,
    "HipRight": 16,
    "KneeRight": 17,
    "AnkleRight": 18,
    "FootRight": 19,
}

# Ordered list for fallback if names are missing but count is 20
JOINT_ORDER = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]

# ==========================================
# Data Parsing & Feature Extraction
# ==========================================


def load_robust_mat(mat_path):
    """
    Robustly parses the .mat file to extract skeleton data.
    Handles polymorphic structure types (struct array vs cell vs single object).
    Returns:
        numpy.ndarray: Shape (NumFrames, 20, 3) containing X, Y, Z coordinates.
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading MAT file {mat_path}: {e}")
        return None

    if "Video" not in mat._fieldnames:
        return None

    video = mat["Video"]
    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames
    num_frames = getattr(video, "NumFrames", len(frames))

    # Initialize skeleton tensor (T, Joints, 3)
    skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

    # If frames is a single object (e.g. 1 frame video), wrap it
    if not isinstance(frames, (list, np.ndarray)):
        frames = [frames]

    for t, frame in enumerate(frames):
        if t >= num_frames:
            break

        if not hasattr(frame, "Skeleton"):
            continue

        skel = frame.Skeleton

        # Handle cases where Skeleton is empty/NaN (no user tracked)
        if skel is None or (
            isinstance(skel, (float, int, np.number)) and np.isnan(skel)
        ):
            # If we have previous frame data, carry it forward (simple interpolation)
            if t > 0:
                skeleton_data[t] = skeleton_data[t - 1]
            continue

        # Check if skel is iterable (array of joints) or single struct
        joints_list = []
        if isinstance(skel, (list, np.ndarray)):
            if len(skel) > 0:
                # Check if it's an array of structs or just values
                if hasattr(skel[0], "WorldPosition"):
                    joints_list = skel
        elif hasattr(skel, "WorldPosition"):
            # Single struct, maybe containing array? Or just one joint?
            # Usually in this dataset, Skeleton is an array of structs.
            # If squeeze_me=True, a struct array might become a numpy array of objects
            pass

        # If we couldn't easily identify list, try iterating if it's a numpy array
        if len(joints_list) == 0 and isinstance(skel, np.ndarray):
            joints_list = skel

        # Parse joints
        valid_frame_data = False

        # Strategy: If we have 20 items, assume order. If we have names, use map.
        if len(joints_list) == 20:
            for i, joint in enumerate(joints_list):
                if hasattr(joint, "WorldPosition"):
                    pos = joint.WorldPosition
                    # Position might be struct with X,Y,Z or array
                    if hasattr(pos, "X") and hasattr(pos, "Y") and hasattr(pos, "Z"):
                        skeleton_data[t, i, 0] = pos.X
                        skeleton_data[t, i, 1] = pos.Y
                        skeleton_data[t, i, 2] = pos.Z
                        valid_frame_data = True
                    elif isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                        skeleton_data[t, i, :] = pos[:3]
                        valid_frame_data = True
        else:
            # Try to find joints by name if structure is different
            # This part is a fallback for complex nested structures
            pass

        # Fallback for missing tracking in this frame
        if not valid_frame_data and t > 0:
            skeleton_data[t] = skeleton_data[t - 1]

    return skeleton_data


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from positions.
    Args:
        positions: (T, 20, 3) numpy array (mm)
    Returns:
        features: (T, 180) numpy array (flattened pos+vel+acc in meters)
    """
    # Convert mm to meters
    pos_m = positions / 1000.0

    # Velocity: dP/dt
    # Pad first frame with 0
    vel = np.zeros_like(pos_m)
    vel[1:] = pos_m[1:] - pos_m[:-1]

    # Acceleration: dV/dt
    acc = np.zeros_like(pos_m)
    acc[1:] = vel[1:] - vel[:-1]

    # Concatenate: (T, 20, 9) -> (T, 180)
    features = np.concatenate([pos_m, vel, acc], axis=2)
    features = features.reshape(features.shape[0], -1)

    return features.astype(np.float32)


def extract_audio_features(audio_path, num_frames):
    """
    Extracts MFCC features and aligns them to video frames.
    Args:
        audio_path: Path to .wav file
        num_frames: Target number of video frames
    Returns:
        mfcc_aligned: (NumFrames, 13) numpy array
    """
    target_dim = 13  # Standard MFCC

    if not os.path.exists(audio_path):
        return np.zeros((num_frames, target_dim), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Compute MFCC
        # We want to align with video.
        # Video FPS ~20. Audio SR ~16k-48k.
        # We'll compute high-res MFCC and interpolate.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=target_dim,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channel, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # Average over channels if stereo -> (n_mfcc, time)

        # Interpolate to match num_frames
        # Input to interpolate needs to be (Batch, Channels, Time)
        mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)

        mfcc_aligned = F.interpolate(
            mfcc, size=num_frames, mode="linear", align_corners=False
        )

        # Transpose to (NumFrames, n_mfcc)
        mfcc_aligned = mfcc_aligned.squeeze(0).transpose(0, 1)
        return mfcc_aligned.numpy()

    except Exception as e:
        # print(f"Audio processing error {audio_path}: {e}")
        return np.zeros((num_frames, target_dim), dtype=np.float32)


# ==========================================
# Data Loading & Caching
# ==========================================


def load_data(mode="train", load_cached_data=True):
    """
    Main data loading function with caching.
    Args:
        mode: 'train', 'val', or 'test'
        load_cached_data: Boolean to use cache
    Returns:
        X: List of feature arrays [(T, InputDim), ...]
        Y: List of label arrays [(T,), ...] (Empty for test)
        sample_ids: List of sample IDs
    """
    cache_file = os.path.join(config.CACHE_DIR, f"dataset_{mode}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            # Reconstruct lists from object arrays
            X = list(data["X"])
            Y = list(data["Y"])
            sample_ids = list(data["sample_ids"])
            print(f"Loaded {mode} data from cache: {len(X)} samples.")
            return X, Y, sample_ids
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch
    metadata_path = os.path.join(config.METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if config.DEBUG:
        df = df.head(config.SUBSET_SIZE)
        print(f"DEBUG MODE: Processing subset of {len(df)} samples.")

    X_list = []
    Y_list = []
    ids_list = []

    print(f"Processing {mode} data...")

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # 1. Skeleton Features
        skeleton_pos = load_robust_mat(mat_path)
        if skeleton_pos is None:
            # Skip corrupted samples or handle gracefully
            # For this challenge, we assume data exists or we pad
            # If completely failed, create dummy data based on audio or just zeros
            # We'll skip to avoid breaking training
            continue

        num_frames = skeleton_pos.shape[0]
        kinematics = compute_kinematics(skeleton_pos)  # (T, 180)

        # 2. Audio Features
        audio_feats = extract_audio_features(audio_path, num_frames)  # (T, 13)

        # 3. Early Fusion
        # Ensure lengths match exactly
        min_len = min(kinematics.shape[0], audio_feats.shape[0])
        features = np.concatenate([kinematics[:min_len], audio_feats[:min_len]], axis=1)

        # 4. Labels (Frame-wise)
        labels = np.zeros(min_len, dtype=np.int64)  # Default 0 (background)

        if mode != "test":
            label_info = json.loads(row["labels"])
            for l in label_info:
                gid = l["id"]
                start = max(0, l["begin"] - 1)  # 1-based to 0-based
                end = min(min_len, l["end"])
                if start < end:
                    labels[start:end] = gid

        X_list.append(features.astype(np.float32))
        Y_list.append(labels)
        ids_list.append(sample_id)

    # 3. Save Cache
    # Use object array for ragged lists
    X_arr = np.array(X_list, dtype=object)
    Y_arr = np.array(Y_list, dtype=object)
    ids_arr = np.array(ids_list, dtype=object)

    np.savez_compressed(cache_file, X=X_arr, Y=Y_arr, sample_ids=ids_arr)

    print(f"Processed and cached {len(X_list)} samples for {mode}.")
    return X_list, Y_list, ids_list


# ==========================================
# Loss & Evaluation
# ==========================================


def truncated_mse_loss(log_probs, threshold=1.0):
    """
    Log-space smoothing loss.
    Penalizes large changes in log-probabilities between adjacent frames.
    Args:
        log_probs: (Batch, Time, Classes)
        threshold: Truncation threshold
    """
    # Diff between t and t-1
    diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]
    squared_diff = diff**2

    # Truncate
    truncated_diff = torch.clamp(squared_diff, max=threshold**2)

    return truncated_diff.mean()


def rle_encode(predictions):
    """
    Run-Length Encoding for predictions.
    Collapses consecutive duplicates and removes background class (0).
    Args:
        predictions: List or 1D array of class IDs
    Returns:
        List of gesture IDs
    """
    if len(predictions) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = [predictions[0]]
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i - 1]:
            collapsed.append(predictions[i])

    # Filter out background (0)
    final_gestures = [g for g in collapsed if g != 0]

    return final_gestures
