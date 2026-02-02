import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
from scipy.spatial.transform import Rotation as R
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
JOINTS_MAP = {
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

# ==========================================
# Core Processing Functions
# ==========================================


def load_polymorphic_mat(mat_path):
    """
    Parses .mat file robustly handling polymorphic Skeleton structures.
    Returns: numpy array of shape (T, 20, 3)
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        # Fallback for corrupted files: return empty
        print(f"Error loading mat file {mat_path}: {e}")
        return None

    if "Video" not in mat:
        return None

    video = mat["Video"]
    # Unwrap 0-d array if squeeze_me=True collapsed it
    if isinstance(video, np.ndarray) and video.ndim == 0:
        video = video.item()

    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames
    num_frames = len(frames) if isinstance(frames, (list, np.ndarray)) else 1

    # Initialize skeleton array (T, Joints, 3)
    skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

    # Track validity for forward fill
    last_valid_frame = np.zeros((20, 3), dtype=np.float32)
    has_valid_data = False

    # Ensure frames is iterable
    if not isinstance(frames, (list, np.ndarray)):
        frames = [frames]

    for t, frame in enumerate(frames):
        # Check if Skeleton exists
        if not hasattr(frame, "Skeleton"):
            skeleton_data[t] = last_valid_frame
            continue

        skel_obj = frame.Skeleton

        # Handle polymorphic Skeleton types
        joints_list = []

        # Case 1: Empty or scalar 0
        if isinstance(skel_obj, (int, float)) or skel_obj is None:
            pass  # No data
        # Case 2: Array of skeletons (multiple users) -> Pick first
        elif isinstance(skel_obj, (list, np.ndarray)) and len(skel_obj) > 0:
            # Usually the first one is the main subject or we should check UserIndex
            # For simplicity and robustness, take the first non-empty one
            joints_list = (
                skel_obj[0] if isinstance(skel_obj[0], (list, np.ndarray)) else skel_obj
            )
        # Case 3: Single Skeleton object
        elif isinstance(skel_obj, scipy.io.matlab.mat_struct):
            joints_list = skel_obj
        else:
            pass  # Unknown type

        # Now extract joints from joints_list
        # It might be an array of joint structs
        current_frame_joints = np.zeros((20, 3), dtype=np.float32)
        joints_found = False

        if isinstance(joints_list, (list, np.ndarray)):
            # Iterate over joints
            for joint in joints_list:
                if hasattr(joint, "JointsType") and hasattr(joint, "WorldPosition"):
                    j_type = str(joint.JointsType).strip()
                    if j_type in JOINTS_MAP:
                        idx = JOINTS_MAP[j_type]
                        pos = joint.WorldPosition
                        # WorldPosition might be struct with X,Y,Z or array
                        if (
                            hasattr(pos, "X")
                            and hasattr(pos, "Y")
                            and hasattr(pos, "Z")
                        ):
                            current_frame_joints[idx] = [pos.X, pos.Y, pos.Z]
                            joints_found = True
                        elif isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                            current_frame_joints[idx] = pos[:3]
                            joints_found = True
        elif isinstance(joints_list, scipy.io.matlab.mat_struct):
            # Sometimes it's a single struct with fields? Unlikely based on prompt.
            # But if joints_list IS the joint (unlikely), handle here.
            pass

        if joints_found:
            skeleton_data[t] = current_frame_joints
            last_valid_frame = current_frame_joints
            has_valid_data = True
        else:
            # Forward fill
            skeleton_data[t] = last_valid_frame

    # If the start was missing, backward fill
    if has_valid_data:
        for t in range(num_frames):
            if np.all(skeleton_data[t] == 0):
                # Find next valid
                for next_t in range(t + 1, num_frames):
                    if not np.all(skeleton_data[next_t] == 0):
                        skeleton_data[t] = skeleton_data[next_t]
                        break

    return skeleton_data


def extract_audio_features(audio_path, num_video_frames):
    """
    Extracts MFCC features aligned to video frames.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        # Return zeros if audio missing
        return np.zeros((num_video_frames, Config.AUDIO_INPUT_DIM), dtype=np.float32)

    # Calculate hop length to match video frames
    # Audio samples = waveform.shape[1]
    # We want num_video_frames output columns
    num_samples = waveform.shape[1]
    if num_video_frames > 0:
        hop_length = int(num_samples / num_video_frames)
        if hop_length < 1:
            hop_length = 1
    else:
        hop_length = 512  # Default

    # MFCC extraction
    # n_mfcc should match Config
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=Config.AUDIO_INPUT_DIM,
        melkwargs={"n_fft": 2048, "hop_length": hop_length, "n_mels": 64},
    )

    mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, Time)
    mfcc = mfcc.mean(dim=0).transpose(0, 1).detach().cpu().numpy()  # (Time, n_mfcc)

    # Align length exactly
    current_frames = mfcc.shape[0]
    if current_frames < num_video_frames:
        # Pad
        padding = np.zeros(
            (num_video_frames - current_frames, Config.AUDIO_INPUT_DIM),
            dtype=np.float32,
        )
        mfcc = np.concatenate([mfcc, padding], axis=0)
    elif current_frames > num_video_frames:
        # Trim
        mfcc = mfcc[:num_video_frames, :]

    return mfcc


def augment_skeleton(positions):
    """
    Applies random Y-axis rotation and scaling.
    positions: (T, 20, 3)
    """
    T, J, C = positions.shape

    # 1. Random Scaling (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)
    aug_pos = positions * scale

    # 2. Random Rotation around Y-axis (-30 to +30 degrees)
    angle_deg = np.random.uniform(-30, 30)
    angle_rad = np.radians(angle_deg)

    # Rotation matrix for Y-axis
    # [ cos  0  sin]
    # [  0   1   0 ]
    # [-sin  0  cos]
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_matrix = np.array(
        [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32
    )

    # Reshape for matmul: (T*J, 3)
    flat_pos = aug_pos.reshape(-1, 3)
    rotated_flat = flat_pos @ rot_matrix.T
    aug_pos = rotated_flat.reshape(T, J, 3)

    return aug_pos


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration.
    positions: (T, 20, 3)
    Returns: (T, 20, 9) -> [Pos, Vel, Acc]
    """
    # Velocity: P_t - P_{t-1}
    # Pad first frame with 0 velocity (replicate position)
    padded_pos = np.pad(positions, ((1, 0), (0, 0), (0, 0)), mode="edge")
    velocity = np.diff(padded_pos, axis=0)  # (T, 20, 3)

    # Acceleration: V_t - V_{t-1}
    padded_vel = np.pad(velocity, ((1, 0), (0, 0), (0, 0)), mode="edge")
    acceleration = np.diff(padded_vel, axis=0)  # (T, 20, 3)

    # Concatenate: (T, 20, 9)
    kinematics = np.concatenate([positions, velocity, acceleration], axis=2)
    return kinematics


def get_feature_vector(skeleton_raw, audio_features, augment=False):
    """
    Combines raw data into final model input.
    skeleton_raw: (T, 20, 3)
    audio_features: (T, 13)
    augment: bool
    Returns: (T, 193)
    """
    # 1. Augmentation
    if augment:
        skel = augment_skeleton(skeleton_raw)
    else:
        skel = skeleton_raw

    # 2. Kinematics
    kinematics = compute_kinematics(skel)  # (T, 20, 9)

    # 3. Flatten Skeleton
    T = kinematics.shape[0]
    skel_flat = kinematics.reshape(T, -1)  # (T, 180)

    # 4. Concatenate Audio
    # Ensure audio matches T (should be handled by loader, but safety check)
    if audio_features.shape[0] != T:
        # Resize audio simply
        # (This is rare if loader works, simple truncation/pad fallback)
        if audio_features.shape[0] > T:
            aud = audio_features[:T]
        else:
            pad = np.zeros((T - audio_features.shape[0], audio_features.shape[1]))
            aud = np.concatenate([audio_features, pad], axis=0)
    else:
        aud = audio_features

    features = np.concatenate([skel_flat, aud], axis=1)  # (T, 193)
    return features.astype(np.float32)


# ==========================================
# Dataset Loading & Caching
# ==========================================


def load_dataset_and_cache(csv_path, split_name, load_cached=True):
    """
    Loads dataset from metadata CSV, processing raw files and caching the result.
    split_name: 'train', 'val', or 'test' (used for filename)
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"dataset_{split_name}.npz")

    # 1. Try Load Cache
    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached {split_name} dataset from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            # Reconstruct list of dicts
            dataset = []
            ids = data["ids"]
            skeletons = data["skeletons"]
            audios = data["audios"]
            labels = data["labels"]

            for i in range(len(ids)):
                dataset.append(
                    {
                        "sample_id": str(ids[i]),
                        "skeleton": skeletons[i],
                        "audio": audios[i],
                        "labels": labels[i],  # This is a list of dicts (object)
                    }
                )
            return dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} dataset from raw files...")
    df = pd.read_csv(csv_path)

    dataset_ids = []
    dataset_skeletons = []
    dataset_audios = []
    dataset_labels = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Parse Labels
        lbls = json.loads(row["labels"]) if isinstance(row["labels"], str) else []

        # Load Skeleton
        skeleton = load_polymorphic_mat(data_path)
        if skeleton is None:
            # Skip corrupted samples or handle?
            # If test, we must provide something. If train, skip.
            if split_name == "test":
                # Dummy data for test if missing
                # Guess length from audio or default
                skeleton = np.zeros((100, 20, 3), dtype=np.float32)
            else:
                continue

        num_frames = skeleton.shape[0]

        # Load Audio
        audio = extract_audio_features(audio_path, num_frames)

        dataset_ids.append(sample_id)
        dataset_skeletons.append(skeleton)
        dataset_audios.append(audio)
        dataset_labels.append(lbls)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")

    # 3. Save Cache
    # Use object array for variable length sequences
    dataset_skeletons_arr = np.array(dataset_skeletons, dtype=object)
    dataset_audios_arr = np.array(dataset_audios, dtype=object)
    dataset_labels_arr = np.array(dataset_labels, dtype=object)

    np.savez_compressed(
        cache_file,
        ids=dataset_ids,
        skeletons=dataset_skeletons_arr,
        audios=dataset_audios_arr,
        labels=dataset_labels_arr,
    )
    print(f"Saved {split_name} dataset to {cache_file}")

    # Return structure
    dataset = []
    for i in range(len(dataset_ids)):
        dataset.append(
            {
                "sample_id": dataset_ids[i],
                "skeleton": dataset_skeletons[i],
                "audio": dataset_audios[i],
                "labels": dataset_labels[i],
            }
        )

    return dataset
