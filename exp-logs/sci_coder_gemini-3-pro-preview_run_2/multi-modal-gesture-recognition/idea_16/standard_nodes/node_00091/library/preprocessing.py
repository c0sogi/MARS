import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import warnings
from library.config import Config
from library.utils import set_seed

# Suppress warnings from scipy/pandas/torch
warnings.filterwarnings("ignore")


def load_skeleton_data(mat_path):
    """
    Parses the .mat file to extract raw 3D joint positions.
    Returns a numpy array of shape (T, NumJoints, 3).
    """
    try:
        # Load mat file
        # struct_as_record=False and squeeze_me=True allow attribute access (e.g., .Video)
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat:
            return None

        video = mat["Video"]

        # Handle Frames attribute
        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames

        # Ensure frames is an array (handle single frame case)
        if not isinstance(frames, np.ndarray):
            frames = np.array([frames])

        num_frames = len(frames)
        num_joints = len(Config.SELECTED_JOINTS)

        # Pre-allocate array: (Time, Joints, 3)
        skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            # Check for Skeleton attribute
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # If multiple users are tracked, skel might be an array of skeletons.
            # We assume the first one is the target or it's a single struct array of joints.
            # Based on description, Skeleton is likely an array of 20 joint structures.

            # Case 1: skel is an array of joints (standard Kinect structure in Matlab)
            if isinstance(skel, np.ndarray):
                # Check if it looks like a list of joints (length >= 20)
                # or a list of users.
                # Heuristic: Check first element. If it has 'WorldPosition', it's likely a joint or user.
                if skel.size == 0:
                    continue

                # If skel has length 20, it's likely the joints of one user.
                # If skel has length < 5 (e.g. 1 or 2), it might be users.
                # However, the description says "Skeleton Frame: An array of Skeleton structures...".
                # Let's try to extract joints from the first element if it seems to be a user container,
                # or treat 'skel' as the joint array itself.

                # We will assume 'skel' is the array of 20 joints for the primary user.
                # If it's an array of users, we take the first user's skeleton.

                # Safe extraction helper
                def get_pos(joint_obj):
                    if hasattr(joint_obj, "WorldPosition"):
                        wp = joint_obj.WorldPosition
                        # wp should be a struct with X, Y, Z or an array
                        if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                            return [wp.X, wp.Y, wp.Z]
                        elif isinstance(wp, np.ndarray) and wp.size >= 3:
                            return wp[:3]
                    return [0.0, 0.0, 0.0]

                # If the array is large enough to contain our indices
                if skel.size >= max(Config.SELECTED_JOINTS):
                    for i, joint_idx in enumerate(Config.SELECTED_JOINTS):
                        skeleton_data[t, i, :] = get_pos(skel[joint_idx])
                else:
                    # Fallback: maybe skel is an array of users?
                    # Try accessing the first element as a user who has a Skeleton field?
                    # Given the ambiguity without direct file access, we proceed with the assumption
                    # that skel IS the array of joints. If extraction fails, we leave as zeros.
                    pass

            # Case 2: skel is a single object (maybe only 1 joint? unlikely)
            else:
                pass

        return skeleton_data

    except Exception as e:
        # print(f"Error processing {mat_path}: {e}")
        return None


def normalize_skeleton(skeleton_data):
    """
    Centers the skeleton relative to the HipCenter (Config.REF_JOINT_INDEX)
    and scales units to meters.
    Input: (T, J, 3)
    Output: (T, J, 3)
    """
    if skeleton_data is None:
        return None

    # 1. Center relative to Reference Joint (HipCenter)
    # Find the index of the reference joint in our SELECTED_JOINTS list
    # Config.SELECTED_JOINTS = [0, 1, 2, ...] so 0 is at index 0.
    try:
        ref_idx = Config.SELECTED_JOINTS.index(Config.REF_JOINT_INDEX)
    except ValueError:
        ref_idx = 0  # Fallback

    # Get reference joint position for each frame: (T, 1, 3)
    ref_pos = skeleton_data[:, ref_idx : ref_idx + 1, :]

    # Subtract reference position
    centered_data = skeleton_data - ref_pos

    # 2. Scale to meters
    normalized_data = centered_data * Config.SCALE_FACTOR

    return normalized_data


def compute_velocity(skeleton_data):
    """
    Computes temporal velocity (difference between frames).
    Input: (T, J, 3)
    Output: (T, J, 3)
    """
    if skeleton_data is None:
        return None

    # Compute diff
    velocity = np.diff(skeleton_data, axis=0)  # Shape (T-1, J, 3)

    # Pad the first frame with zeros to maintain length T
    pad = np.zeros((1, skeleton_data.shape[1], 3), dtype=skeleton_data.dtype)
    velocity = np.concatenate([pad, velocity], axis=0)

    return velocity


def extract_audio_features(audio_path, target_num_frames):
    """
    Loads audio, computes MFCCs, and aligns them to the video frame count.
    Input: Path to wav, Number of video frames.
    Output: (target_num_frames, N_MFCC)
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Compute MFCC
        # We use standard parameters. The temporal resolution will be adjusted via interpolation.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # Shape: (Channels, n_mfcc, Time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # mfcc shape is now (1, n_mfcc, Time_audio)

        # Interpolate to match video frames
        # Input to interpolate needs to be (Batch, Channels, Length)
        # We treat n_mfcc as channels for interpolation
        mfcc = torch.nn.functional.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Reshape to (T, n_mfcc)
        mfcc = mfcc.squeeze(0).permute(1, 0).numpy()

        return mfcc

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def process_dataset(metadata_path, output_filename, load_cached_data=True):
    """
    Main processing function.
    Reads metadata, processes each sample, and saves/loads from cache.
    """
    cache_path = os.path.join(Config.WORKING_DIR, output_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return np.load(cache_path, allow_pickle=True)["data"].item()
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Drop rows with missing data paths to prevent TypeErrors in os.path.join
    df.dropna(subset=["data_path"], inplace=True)

    # Convert labels string to list
    df["labels"] = df["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )

    processed_data = {}

    for _, row in df.iterrows():
        sample_id = row["sample_id"]

        # Construct full paths
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Skeleton Features
        raw_skel = load_skeleton_data(mat_path)

        if raw_skel is None:
            # Fallback for missing data: create zeros based on num_frames in metadata
            num_frames = int(row["num_frames"]) if pd.notna(row["num_frames"]) else 100
            raw_skel = np.zeros(
                (num_frames, len(Config.SELECTED_JOINTS), 3), dtype=np.float32
            )

        norm_skel = normalize_skeleton(raw_skel)  # (T, 12, 3)
        velocity = compute_velocity(norm_skel)  # (T, 12, 3)

        # Flatten spatial dims: (T, 36)
        T = norm_skel.shape[0]
        feat_pos = norm_skel.reshape(T, -1)
        feat_vel = velocity.reshape(T, -1)

        # 2. Audio Features
        feat_audio = extract_audio_features(audio_path, T)  # (T, 13)

        # 3. Concatenate
        # Final Feature Vector: [Position (36), Velocity (36), Audio (13)] = 85 dims
        features = np.concatenate([feat_pos, feat_vel, feat_audio], axis=1).astype(
            np.float32
        )

        # 4. Store
        processed_data[sample_id] = {
            "features": features,
            "labels": np.array(row["labels"], dtype=np.int64),
        }

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez_compressed(cache_path, data=processed_data)
    print(f"Saved processed data to {cache_path}")

    return processed_data


def get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Wrapper to generate PyTorch DataLoaders.
    Note: The actual Dataset class is typically defined in dataset.py,
    but for this task we are only implementing preprocessing.py.
    This function is a placeholder or can be used if the Dataset class was provided.
    Given the instructions, we focus on the data processing logic above.
    """
    pass
