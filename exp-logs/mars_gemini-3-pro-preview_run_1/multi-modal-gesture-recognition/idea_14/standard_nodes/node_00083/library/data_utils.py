import os
import numpy as np
import scipy.io
import torch
import torchaudio
import pandas as pd
import warnings
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    CACHE_DIR,
    FPS,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MFCC,
    SKELETON_JOINTS,
    SKELETON_CHANNELS,
    SKELETON_INPUT_SIZE,
    AUDIO_INPUT_SIZE,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def load_mat_file(rel_path):
    """
    Parses the .mat file to extract Skeleton data.
    Returns: numpy array of shape (T, 20, 3) or None if invalid.
    """
    if not isinstance(rel_path, str):
        return None

    full_path = os.path.join(INPUT_DIR, rel_path)
    try:
        # Load mat file
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat:
            return None

        video = mat["Video"]
        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames

        # Handle cases where Frames might be a single object or empty
        if isinstance(frames, np.ndarray):
            if frames.size == 0:
                return None
            # If it's a 0-d array (scalar object), wrap it
            if frames.ndim == 0:
                frames = np.array([frames])
        elif not isinstance(frames, (list, np.ndarray)):
            # Single frame object
            frames = np.array([frames])

        num_frames = len(frames)
        skeleton_data = np.zeros(
            (num_frames, SKELETON_JOINTS, SKELETON_CHANNELS), dtype=np.float32
        )

        for i, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Handle multiple users: take the first one if array
            if isinstance(skel, np.ndarray):
                if skel.size == 0:
                    continue
                skel = skel[0]  # Assume first user is the target

            if not hasattr(skel, "WorldPosition"):
                continue

            wp = skel.WorldPosition

            # Extract XYZ
            # Case 1: WorldPosition is a struct with X, Y, Z fields
            if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                # X, Y, Z can be scalars or arrays (if multiple joints)
                # Assuming they are arrays of shape (20,) or (20,1)
                x = np.atleast_1d(wp.X).flatten()
                y = np.atleast_1d(wp.Y).flatten()
                z = np.atleast_1d(wp.Z).flatten()

                # Ensure we have 20 joints
                if len(x) == SKELETON_JOINTS:
                    skeleton_data[i, :, 0] = x
                    skeleton_data[i, :, 1] = y
                    skeleton_data[i, :, 2] = z

            # Case 2: WorldPosition is a matrix (e.g. 20x3 or 3x20)
            elif isinstance(wp, np.ndarray):
                if wp.shape == (SKELETON_JOINTS, 3):
                    skeleton_data[i] = wp
                elif wp.shape == (3, SKELETON_JOINTS):
                    skeleton_data[i] = wp.T

        return skeleton_data

    except Exception as e:
        return None


def compute_skeleton_features(raw_skeleton):
    """
    Computes Root-Relative coordinates and flattens.
    Input: (T, 20, 3)
    Output: (T, 60)
    """
    if raw_skeleton is None:
        return None

    # HipCenter is index 0 based on the prompt list
    # 0: HipCenter
    # Substract HipCenter from all joints
    root = raw_skeleton[:, 0:1, :]  # (T, 1, 3)
    relative = raw_skeleton - root  # (T, 20, 3)

    # Flatten
    T = relative.shape[0]
    features = relative.reshape(T, -1)  # (T, 60)

    return features


def compute_audio_features(rel_path, num_video_frames):
    """
    Computes MFCCs aligned with video frames.
    Input: Audio path, target number of frames
    Output: (T, N_MFCC)
    """
    if rel_path is None or pd.isna(rel_path):
        return np.zeros((num_video_frames, N_MFCC), dtype=np.float32)

    full_path = os.path.join(INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        return np.zeros((num_video_frames, N_MFCC), dtype=np.float32)

    try:
        # Load audio
        waveform, sr = torchaudio.load(full_path)

        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Mix to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute MFCC
        # hop_length=800 ensures roughly 20Hz output (16000/800 = 20)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "center": True,
                "power": 2.0,
            },
        )

        mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, T_audio)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (T_audio, n_mfcc)

        # Convert to numpy
        mfcc = mfcc.detach().numpy()

        # Align length
        t_audio = mfcc.shape[0]
        if t_audio < num_video_frames:
            # Pad
            pad_width = num_video_frames - t_audio
            # Pad with zeros at the end
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
        elif t_audio > num_video_frames:
            # Truncate
            mfcc = mfcc[:num_video_frames, :]

        return mfcc.astype(np.float32)

    except Exception as e:
        return np.zeros((num_video_frames, N_MFCC), dtype=np.float32)


def process_sample(row, load_cached_data=True):
    """
    Processes a single sample: loads raw data, extracts features, caches result.
    Args:
        row: pandas Series or dict containing 'sample_id', 'data_path', 'audio_path', 'labels'
        load_cached_data: bool
    Returns:
        dict with 'skeleton', 'audio', 'labels', 'sample_id'
    """
    sample_id = row["sample_id"]
    cache_path = os.path.join(CACHE_DIR, f"{sample_id}.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "skeleton": data["skeleton"],
                "audio": data["audio"],
                "labels": data["labels"],
                "sample_id": str(data["sample_id"]),
            }
        except Exception:
            pass  # Fallback to compute

    # 2. Compute
    # Load Skeleton
    raw_skel = load_mat_file(row["data_path"])
    if raw_skel is None:
        # If skeleton is missing, we can't do much. Return None to filter this sample.
        return None

    skel_features = compute_skeleton_features(raw_skel)
    num_frames = skel_features.shape[0]

    # Load Audio
    audio_features = compute_audio_features(row["audio_path"], num_frames)

    # Process Labels
    # Labels are string "1,2,3" or empty
    label_str = str(row["labels"]) if pd.notna(row["labels"]) else ""
    if label_str.strip() == "":
        labels = np.array([], dtype=np.int64)
    else:
        try:
            labels = np.array([int(x) for x in label_str.split(",")], dtype=np.int64)
        except ValueError:
            labels = np.array([], dtype=np.int64)

    # 3. Save Cache
    np.savez_compressed(
        cache_path,
        skeleton=skel_features,
        audio=audio_features,
        labels=labels,
        sample_id=sample_id,
    )

    return {
        "skeleton": skel_features,
        "audio": audio_features,
        "labels": labels,
        "sample_id": sample_id,
    }


def compute_global_stats(df, load_cached_data=True):
    """
    Computes global mean and std for Skeleton and Audio features.
    Args:
        df: pandas DataFrame of training data
        load_cached_data: bool
    Returns:
        stats_dict: {'skel_mean': ..., 'skel_std': ..., 'audio_mean': ..., 'audio_std': ...}
    """
    stats_path = os.path.join(WORKING_DIR, "stats.npz")

    if load_cached_data and os.path.exists(stats_path):
        try:
            stats = np.load(stats_path)
            return {
                "skel_mean": stats["skel_mean"],
                "skel_std": stats["skel_std"],
                "audio_mean": stats["audio_mean"],
                "audio_std": stats["audio_std"],
            }
        except Exception:
            pass

    # Compute from scratch
    skel_sum = np.zeros(SKELETON_INPUT_SIZE)
    skel_sq_sum = np.zeros(SKELETON_INPUT_SIZE)
    skel_count = 0

    audio_sum = np.zeros(AUDIO_INPUT_SIZE)
    audio_sq_sum = np.zeros(AUDIO_INPUT_SIZE)
    audio_count = 0

    # Iterate through all samples in DF
    # We use process_sample to ensure we get the processed features
    # We force load_cached_data=True to use individual caches if available
    for idx, row in df.iterrows():
        sample = process_sample(row, load_cached_data=True)
        if sample is None:
            continue

        s = sample["skeleton"]
        a = sample["audio"]

        if s is not None and s.size > 0:
            skel_sum += np.sum(s, axis=0)
            skel_sq_sum += np.sum(s**2, axis=0)
            skel_count += s.shape[0]

        if a is not None and a.size > 0:
            audio_sum += np.sum(a, axis=0)
            audio_sq_sum += np.sum(a**2, axis=0)
            audio_count += a.shape[0]

    # Calculate Mean and Std
    skel_mean = skel_sum / max(1, skel_count)
    skel_std = np.sqrt(np.maximum(0, (skel_sq_sum / max(1, skel_count)) - skel_mean**2))
    # Avoid div by zero in normalization later
    skel_std[skel_std < 1e-6] = 1.0

    audio_mean = audio_sum / max(1, audio_count)
    audio_std = np.sqrt(
        np.maximum(0, (audio_sq_sum / max(1, audio_count)) - audio_mean**2)
    )
    audio_std[audio_std < 1e-6] = 1.0

    # Save
    np.savez(
        stats_path,
        skel_mean=skel_mean,
        skel_std=skel_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )

    return {
        "skel_mean": skel_mean,
        "skel_std": skel_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }
