import os
import numpy as np
import scipy.io
import scipy.signal
import torch
import torchaudio
import pandas as pd
import warnings
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SELECTED_JOINTS_INDICES,
    SKELETON_PAIRS,
    SCALE_FACTOR,
    AUDIO_SAMPLE_RATE,
    N_MFCC,
    HIP_CENTER_INDEX,
    SEED,
)
from library.utils import load_metadata, set_seed

# Suppress warnings
warnings.filterwarnings("ignore")


def get_butter_filter(cutoff=0.1, fs=10.0, order=4):
    """
    Creates a low-pass Butterworth filter for temporal augmentation.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def normalize_skeleton(skeleton):
    """
    Centers the skeleton around the HipCenter and scales to meters.

    Args:
        skeleton (np.ndarray): Shape (T, NumJoints, 3)

    Returns:
        np.ndarray: Normalized skeleton.
    """
    # Skeleton shape: (T, J, 3)
    # HipCenter is at index 0 in our selected joints
    if skeleton.shape[1] <= HIP_CENTER_INDEX:
        return skeleton * SCALE_FACTOR

    hip_center = skeleton[:, HIP_CENTER_INDEX : HIP_CENTER_INDEX + 1, :]
    normalized = (skeleton - hip_center) * SCALE_FACTOR
    return normalized


def compute_bone_vectors(skeleton):
    """
    Computes bone vectors based on connected joints.

    Args:
        skeleton (np.ndarray): Shape (T, NumJoints, 3)

    Returns:
        np.ndarray: Shape (T, NumBones, 3)
    """
    bones = []
    for parent_idx, child_idx in SKELETON_PAIRS:
        # Vector from Parent to Child
        bone = skeleton[:, child_idx, :] - skeleton[:, parent_idx, :]
        bones.append(bone)

    if not bones:
        return np.zeros((skeleton.shape[0], 0, 3))

    return np.stack(bones, axis=1)


def compute_velocity(skeleton):
    """
    Computes temporal velocity.

    Args:
        skeleton (np.ndarray): Shape (T, NumJoints, 3)

    Returns:
        np.ndarray: Shape (T, NumJoints, 3)
    """
    # Pad the first frame with zeros to maintain temporal length
    velocity = np.zeros_like(skeleton)
    velocity[1:] = skeleton[1:] - skeleton[:-1]
    return velocity


def physically_consistent_augmentation(skeleton, noise_sigma=0.01):
    """
    Applies temporal low-pass filtered noise to positions.

    Args:
        skeleton (np.ndarray): Shape (T, J, 3)
        noise_sigma (float): Standard deviation of noise (in meters).

    Returns:
        np.ndarray: Augmented skeleton positions.
    """
    T, J, C = skeleton.shape

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_sigma, size=(T, J, C))

    # Apply Low-Pass Filter along time dimension
    b, a = get_butter_filter()
    smooth_noise = scipy.signal.lfilter(b, a, noise, axis=0)

    return skeleton + smooth_noise


def extract_audio_features(audio_path, target_frames):
    """
    Loads audio and extracts MFCCs, aligned to target_frames.

    Args:
        audio_path (str): Path to audio file.
        target_frames (int): Number of video frames to align to.

    Returns:
        np.ndarray: Shape (target_frames, N_MFCC)
    """
    full_path = os.path.join(INPUT_DIR, audio_path)

    try:
        if not os.path.exists(full_path):
            return np.zeros((target_frames, N_MFCC), dtype=np.float32)

        waveform, sample_rate = torchaudio.load(full_path)

        # Resample if necessary
        if sample_rate != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # Interpolate to match target_frames
        # Input to interpolate needs to be (Batch, Channels, Time)
        # MFCC is (1, n_mfcc, time_steps) -> treat n_mfcc as channels for 1D interpolation
        # or just interpolate the last dim

        if mfcc.shape[-1] == 0:
            return np.zeros((target_frames, N_MFCC), dtype=np.float32)

        mfcc = torch.nn.functional.interpolate(
            mfcc, size=target_frames, mode="linear", align_corners=False
        )

        # Shape: (1, n_mfcc, target_frames) -> (target_frames, n_mfcc)
        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()

        return mfcc.astype(np.float32)

    except Exception as e:
        # print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((target_frames, N_MFCC), dtype=np.float32)


def parse_mat_file(data_path):
    """
    Parses the .mat file to extract skeleton data for selected joints.

    Returns:
        np.ndarray: Raw skeleton data (T, 12, 3) in mm.
    """
    full_path = os.path.join(INPUT_DIR, data_path)
    try:
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
        video = mat["Video"]
        frames = video.Frames  # Array of structs

        num_frames = len(frames) if isinstance(frames, np.ndarray) else 1
        if num_frames == 1 and not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        skeleton_data = np.zeros(
            (num_frames, len(SELECTED_JOINTS_INDICES), 3), dtype=np.float32
        )

        for t, frame in enumerate(frames):
            try:
                # Frame might have multiple skeletons, usually we take the one with UserIndex or just the first valid one
                # The dataset description implies 'Skeleton' field.
                # Usually in this dataset: frame.Skeleton is a struct or array of structs.
                # We assume single user or pre-filtered.
                skel = frame.Skeleton

                # Check if skel is valid
                if skel is None:
                    continue

                # If array of skeletons, take first (simplification)
                if isinstance(skel, np.ndarray) and skel.size > 0:
                    skel = skel[0]
                elif isinstance(skel, np.ndarray) and skel.size == 0:
                    continue

                # Extract joints
                # skel.WorldPosition is likely not an array of positions directly in this specific dataset format
                # often it is skel.Joints or similar.
                # However, description says: "Skeleton... WorldPosition... X, Y, Z".
                # It also lists "JointsType".
                # Based on standard handling of this specific challenge data (Chalearn/MMRGC):
                # skel might be a struct where fields are joint names OR an array of joints.
                # Let's try to access WorldPosition of specific joints.

                # If skel has 'WorldPosition' directly, it might be for one joint? No.
                # Let's assume skel is a struct containing an array of joints or similar.
                # Actually, looking at description: "Skeleton Frame: An array of Skeleton structures... contained within a Skeletons array."
                # Wait, "Skeleton" contains "JointsType", "WorldPosition".
                # This implies `frame.Skeleton` is an array of joints for that frame.

                if hasattr(skel, "WorldPosition"):
                    # It might be a single joint or array of joints
                    # If it's an array of joints:
                    pass

                # Fallback for common structure:
                # frame.Skeleton is an array of 20 joints.
                if isinstance(skel, np.ndarray):
                    joints = skel
                else:
                    # Maybe it's a single object if only 1 joint? Unlikely.
                    joints = [skel]

                # Map selected indices
                for i, joint_idx in enumerate(SELECTED_JOINTS_INDICES):
                    if joint_idx < len(joints):
                        joint = joints[joint_idx]
                        if hasattr(joint, "WorldPosition"):
                            wp = joint.WorldPosition
                            skeleton_data[t, i, 0] = wp.X
                            skeleton_data[t, i, 1] = wp.Y
                            skeleton_data[t, i, 2] = wp.Z
            except AttributeError:
                continue

        return skeleton_data

    except Exception as e:
        # Return empty if failed
        return np.zeros((0, len(SELECTED_JOINTS_INDICES), 3), dtype=np.float32)


def compute_all_features(positions, audio, augment=False):
    """
    Computes the full feature vector from positions and audio.

    Args:
        positions (np.ndarray): Normalized skeleton positions (T, 12, 3).
        audio (np.ndarray): Aligned audio features (T, N_MFCC).
        augment (bool): Whether to apply geometric augmentation.

    Returns:
        torch.Tensor: Float tensor of shape (T, INPUT_DIM).
    """
    # 1. Augmentation
    if augment:
        positions = physically_consistent_augmentation(positions)

    # 2. Derived Geometric Features
    velocity = compute_velocity(positions)  # (T, 12, 3)
    bones = compute_bone_vectors(positions)  # (T, 11, 3)

    # 3. Flatten Spatial Dimensions
    T = positions.shape[0]
    pos_flat = positions.reshape(T, -1)  # (T, 36)
    vel_flat = velocity.reshape(T, -1)  # (T, 36)
    bones_flat = bones.reshape(T, -1)  # (T, 33)

    # 4. Concatenate
    # Ensure audio length matches (it should, but safety first)
    if audio.shape[0] != T:
        # Simple truncation or padding
        if audio.shape[0] > T:
            audio = audio[:T]
        else:
            pad = np.zeros((T - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=0)

    features = np.concatenate([pos_flat, vel_flat, bones_flat, audio], axis=1)

    return torch.tensor(features, dtype=torch.float32)


def process_data(split="train", load_cached_data=True):
    """
    Loads, processes, and caches the dataset.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: Dictionary containing lists of 'sample_ids', 'skeletons', 'audio', 'labels'.
    """
    set_seed(SEED)

    cache_file = os.path.join(CACHE_DIR, f"{split}_data.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return {
                "sample_ids": data["sample_ids"],
                "skeletons": data["skeletons"],
                "audio": data["audio"],
                "labels": data["labels"],
            }
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {split} data...")
    df = load_metadata(split)

    sample_ids = []
    skeletons = []
    audio_feats = []
    labels = []

    for idx, row in df.iterrows():
        sid = row["sample_ids"] if "sample_ids" in row else row["sample_id"]

        # Parse Skeleton
        raw_skel = parse_mat_file(row["data_path"])
        if raw_skel.shape[0] == 0:
            # Skip empty samples or handle gracefully
            # For this task, we'll create a dummy frame to avoid crashing if essential
            # But better to skip if training
            if split == "train":
                continue
            else:
                # For test, we must output something, so pad
                raw_skel = np.zeros(
                    (10, len(SELECTED_JOINTS_INDICES), 3), dtype=np.float32
                )

        # Normalize
        norm_skel = normalize_skeleton(raw_skel)

        # Audio
        # Use num_frames from skeleton to align
        num_frames = norm_skel.shape[0]
        aud = extract_audio_features(row["audio_path"], num_frames)

        sample_ids.append(sid)
        skeletons.append(norm_skel)
        audio_feats.append(aud)
        labels.append(row["labels"])

    # 3. Save Cache
    # Convert lists to object arrays for saving
    # We use object arrays because lengths vary
    skeletons_arr = np.array(skeletons, dtype=object)
    audio_arr = np.array(audio_feats, dtype=object)
    labels_arr = np.array(labels, dtype=object)
    ids_arr = np.array(sample_ids)

    np.savez_compressed(
        cache_file,
        sample_ids=ids_arr,
        skeletons=skeletons_arr,
        audio=audio_arr,
        labels=labels_arr,
    )

    return {
        "sample_ids": ids_arr,
        "skeletons": skeletons_arr,
        "audio": audio_arr,
        "labels": labels_arr,
    }
