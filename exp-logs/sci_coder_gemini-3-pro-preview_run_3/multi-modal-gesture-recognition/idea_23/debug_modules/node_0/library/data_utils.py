import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from library import config

# ==========================================
# Reproducibility
# ==========================================
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# ==========================================
# Helper Functions
# ==========================================


def parse_labels(label_json_str):
    """
    Parses the JSON string containing label data.
    """
    if not isinstance(label_json_str, str):
        return []
    try:
        return json.loads(label_json_str)
    except json.JSONDecodeError:
        return []


def _extract_skeleton_from_frame(frame_obj):
    """
    Robustly extracts 20 joints (x,y,z) from a single frame object.
    Handles polymorphic structure of Skeleton field in .mat files.
    Returns: np.ndarray shape (20, 3) or None if extraction fails.
    """
    if not hasattr(frame_obj, "Skeleton"):
        return None

    skel = frame_obj.Skeleton

    # Case 1: Skeleton is empty or None
    if skel is None:
        return None

    # Container for joints
    joints_data = []

    # Helper to extract XYZ from a position object/struct
    def get_xyz(pos_obj):
        try:
            # Check if it's a struct with X, Y, Z attributes
            if (
                hasattr(pos_obj, "X")
                and hasattr(pos_obj, "Y")
                and hasattr(pos_obj, "Z")
            ):
                return [float(pos_obj.X), float(pos_obj.Y), float(pos_obj.Z)]
            # Check if it's a dictionary-like or array
            elif isinstance(pos_obj, (np.ndarray, list)) and len(pos_obj) >= 3:
                return [float(pos_obj[0]), float(pos_obj[1]), float(pos_obj[2])]
        except:
            pass
        return [0.0, 0.0, 0.0]

    try:
        # Case 2: Skeleton is an array of joints (ndarray or list)
        if isinstance(skel, (np.ndarray, list)):
            if len(skel) == 0:
                return None

            # If it's a list of structs
            for joint in skel:
                if hasattr(joint, "WorldPosition"):
                    joints_data.append(get_xyz(joint.WorldPosition))
                else:
                    # Maybe the joint itself is the position? Unlikely but possible
                    joints_data.append([0.0, 0.0, 0.0])

        # Case 3: Skeleton is a single struct containing arrays or nested structs
        elif isinstance(skel, scipy.io.matlab.mat_struct):
            # Check if it has 'WorldPosition' directly (unlikely for 20 joints)
            # Or if it acts as an array
            pass

        # Validation
        if len(joints_data) == 20:
            return np.array(joints_data, dtype=np.float32)

    except Exception:
        pass

    return None


def resample_features(features, target_len):
    """
    Resample feature matrix to target length using linear interpolation.
    Args:
        features: (T_in, D)
        target_len: int
    Returns:
        (target_len, D)
    """
    if len(features) == 0:
        return np.zeros((target_len, features.shape[1]), dtype=np.float32)

    features_tensor = torch.from_numpy(features).T.unsqueeze(0)  # (1, D, T_in)

    # Use interpolate
    out = torch.nn.functional.interpolate(
        features_tensor, size=target_len, mode="linear", align_corners=False
    )

    return out.squeeze(0).T.numpy()


# ==========================================
# Core Loading Functions
# ==========================================


def load_mat_safe(mat_path):
    """
    Loads skeleton data from .mat file.
    Returns: np.ndarray of shape (num_frames, 20, 3) in meters.
    """
    if not os.path.exists(mat_path):
        # Return a dummy small sequence if file missing (should not happen based on checks)
        return np.zeros((10, 20, 3), dtype=np.float32)

    try:
        # Load mat file
        # struct_as_record=False loads structs as objects with attributes
        # squeeze_me=True simplifies single-element arrays
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        if "Video" not in mat:
            return np.zeros((10, 20, 3), dtype=np.float32)

        video = mat["Video"]

        if not hasattr(video, "Frames"):
            return np.zeros((10, 20, 3), dtype=np.float32)

        frames = video.Frames

        # Handle single frame case (not iterable)
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        num_frames = len(frames)
        skeleton_seq = []

        last_valid_frame = np.zeros((20, 3), dtype=np.float32)

        for f in frames:
            skel_data = _extract_skeleton_from_frame(f)

            if skel_data is not None:
                last_valid_frame = skel_data
                skeleton_seq.append(skel_data)
            else:
                # Use last valid frame (simple imputation)
                skeleton_seq.append(last_valid_frame.copy())

        # Convert to numpy
        skeleton_seq = np.array(skeleton_seq, dtype=np.float32)

        # Normalize: Millimeters -> Meters
        skeleton_seq = skeleton_seq / 1000.0

        # Ensure shape (T, 20, 3)
        if (
            skeleton_seq.ndim != 3
            or skeleton_seq.shape[1] != 20
            or skeleton_seq.shape[2] != 3
        ):
            # Fallback
            return np.zeros((max(1, num_frames), 20, 3), dtype=np.float32)

        return skeleton_seq

    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return np.zeros((10, 20, 3), dtype=np.float32)


def load_audio_mfcc(audio_path, target_num_frames):
    """
    Loads audio, computes MFCC, and aligns to video frame count.
    Returns: np.ndarray (target_num_frames, n_mfcc)
    """
    if not os.path.exists(audio_path):
        return np.zeros((target_num_frames, config.AUDIO_N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary (config assumes 16k)
        if sample_rate != config.AUDIO_SAMPLE_RATE:
            resampler = T.Resample(sample_rate, config.AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Mix to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.AUDIO_N_MFCC,
            melkwargs={
                "n_fft": config.AUDIO_N_FFT,
                "hop_length": config.AUDIO_HOP_LENGTH,
                "n_mels": 64,
                "center": False,
            },
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (time, n_mfcc)

        # Align to video frames
        aligned_mfcc = resample_features(mfcc, target_num_frames)

        return aligned_mfcc

    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return np.zeros((target_num_frames, config.AUDIO_N_MFCC), dtype=np.float32)


# ==========================================
# Dataset Processing & Caching
# ==========================================


def process_dataset(metadata_df, subset_name, load_cached_data=True):
    """
    Loads, processes, and caches the dataset.

    Args:
        metadata_df: DataFrame containing file paths.
        subset_name: 'train', 'val', or 'test'.
        load_cached_data: Boolean flag to use cache.

    Returns:
        Dictionary containing:
        - 'skeleton': List of (T, 20, 3) arrays
        - 'audio': List of (T, 13) arrays
        - 'labels': List of label lists
        - 'sample_ids': List of sample IDs
    """
    cache_path = os.path.join(config.CACHE_DIR, f"dataset_{subset_name}.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {subset_name} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "skeleton": list(data["skeleton"]),
                "audio": list(data["audio"]),
                "labels": list(data["labels"]),
                "sample_ids": list(data["sample_ids"]),
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {subset_name} dataset ({len(metadata_df)} samples)...")

    all_skeletons = []
    all_audio = []
    all_labels = []
    all_ids = []

    for idx, row in metadata_df.iterrows():
        # Construct full paths
        mat_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # Load Skeleton
        skeleton = load_mat_safe(mat_path)  # (T, 20, 3)
        num_frames = skeleton.shape[0]

        # Load Audio (Aligned)
        audio = load_audio_mfcc(audio_path, num_frames)  # (T, 13)

        # Parse Labels
        labels = parse_labels(row["labels"])

        all_skeletons.append(skeleton)
        all_audio.append(audio)
        all_labels.append(labels)
        all_ids.append(row["sample_id"])

    # 3. Save Cache
    # We use object array for variable length sequences
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(
        cache_path,
        skeleton=np.array(all_skeletons, dtype=object),
        audio=np.array(all_audio, dtype=object),
        labels=np.array(all_labels, dtype=object),
        sample_ids=np.array(all_ids, dtype=object),
    )

    return {
        "skeleton": all_skeletons,
        "audio": all_audio,
        "labels": all_labels,
        "sample_ids": all_ids,
    }
