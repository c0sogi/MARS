import os
import numpy as np
import scipy.io
import scipy.interpolate
import torch
import torchaudio
import soundfile as sf
from library import config

# ==========================================
# 1. Robust Skeleton Parsing
# ==========================================


def safe_parse_skeleton(mat_path):
    """
    Parses the .mat file to extract 3D skeleton joint positions.
    Handles polymorphic data types (struct array vs cell array vs primitive)
    to prevent silent failures.

    Args:
        mat_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Shape (T, NumJoints, 3) containing WorldPosition (x,y,z).
                    Returns None if parsing fails completely.
    """
    try:
        # Load mat file with squeeze_me to simplify structure access
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

    # Access top-level variable as dictionary key (Cite debug_lesson_2)
    if "Video" not in mat:
        return None

    video = mat["Video"]

    # Unwrap 0-d numpy array if present (Cite debug_lesson_16)
    if isinstance(video, np.ndarray) and video.ndim == 0:
        video = video.item()

    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames

    # Handle case where Frames is a single object (1 frame video)
    if not isinstance(frames, (list, np.ndarray)):
        frames = [frames]

    num_frames = len(frames)
    num_joints = config.NUM_JOINTS

    # Pre-allocate array: (T, J, 3)
    skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    # Standard Kinect Joint Order (matches dataset description)
    # Used for fallback if JointsType name matching fails
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head, ...

    for t, frame in enumerate(frames):
        # Check if Skeleton field exists and is not empty
        if not hasattr(frame, "Skeleton"):
            continue

        skel = frame.Skeleton

        # Polymorphic check: Skeleton might be:
        # 1. A single struct (one user)
        # 2. An array of structs (multiple users)
        # 3. Empty/NaN (no user)

        target_skel = None

        if isinstance(skel, scipy.io.matlab.mat_struct):
            # Single user
            target_skel = skel
        elif isinstance(skel, (np.ndarray, list)):
            if len(skel) > 0:
                # Multiple users: Strategy - pick the first one or the one with UserIndex
                # For simplicity and robustness, we pick the first valid tracked skeleton
                # In a more complex scenario, we would check UserIndex map.
                if isinstance(skel[0], scipy.io.matlab.mat_struct):
                    target_skel = skel[0]

        # If we found a valid skeleton struct
        if target_skel is not None:
            # The skeleton struct usually contains an array of joints or fields for joints
            # Based on description: "Skeleton Frame... contains joint positions... JointsType"
            # Often in these .mat files, the Skeleton object itself is an array of 20 joint objects
            # OR it has a field 'Joint' which is the array.

            # Let's inspect the target_skel to find joints.
            # Case A: target_skel is an array of 20 structs (the joints themselves)
            # Case B: target_skel has a field 'Joints' or similar.

            # Based on common datasets of this type (MSR-DailyActivity etc),
            # and the description "Structure... JointsType... WorldPosition",
            # it implies the Skeleton variable is an array of Joint structures.

            joints_array = None

            # If target_skel is iterable and has length 20, it's likely the joints array
            if (
                isinstance(target_skel, (np.ndarray, list))
                and len(target_skel) == num_joints
            ):
                joints_array = target_skel
            # If target_skel is a struct, maybe it acts as the array (squeeze_me weirdness)
            # Or we check if it has WorldPosition directly (unlikely for whole body)

            # Fallback for "squeeze_me" collapsing:
            # If the .mat had Skeletons -> Skeleton (array of joints),
            # `target_skel` might be the array of joints.

            if joints_array is None:
                # Try to treat target_skel as the joints array if it has WorldPosition
                # This happens if it's a numpy array of objects
                try:
                    if len(target_skel) == num_joints:
                        joints_array = target_skel
                except:
                    pass

            if joints_array is not None:
                for j_idx in range(min(len(joints_array), num_joints)):
                    joint = joints_array[j_idx]
                    if hasattr(joint, "WorldPosition"):
                        pos = joint.WorldPosition
                        # pos should have X, Y, Z
                        # Handle polymorphic types (Object with .X or Array) Cite debug_lesson_10
                        if hasattr(pos, "X"):
                            skeleton_data[t, j_idx, 0] = pos.X / 1000.0
                            skeleton_data[t, j_idx, 1] = pos.Y / 1000.0
                            skeleton_data[t, j_idx, 2] = pos.Z / 1000.0
                        elif isinstance(pos, (np.ndarray, list)) and len(pos) >= 3:
                            skeleton_data[t, j_idx, 0] = pos[0] / 1000.0
                            skeleton_data[t, j_idx, 1] = pos[1] / 1000.0
                            skeleton_data[t, j_idx, 2] = pos[2] / 1000.0

    # Post-processing: Handle dropped frames / missing skeletons
    # Simple strategy: Forward fill, then backward fill
    # Identify empty frames (all zeros)
    is_empty = np.all(skeleton_data == 0, axis=(1, 2))

    # If all frames are empty, return zeros (silent failure handled by model or ignored)
    if np.all(is_empty):
        return skeleton_data

    # Forward fill
    for t in range(1, num_frames):
        if is_empty[t]:
            skeleton_data[t] = skeleton_data[t - 1]
            is_empty[t] = False  # It's filled now

    # Backward fill (for start of sequence)
    for t in range(num_frames - 2, -1, -1):
        if np.all(skeleton_data[t] == 0):  # Check again as is_empty might be stale
            skeleton_data[t] = skeleton_data[t + 1]

    return skeleton_data


# ==========================================
# 2. Kinematics & Augmentation
# ==========================================


def augment_skeleton(positions):
    """
    Applies random rotation (Y-axis) and scaling to skeleton positions.
    Args:
        positions: (T, J, 3) np.ndarray
    Returns:
        augmented_positions: (T, J, 3) np.ndarray
    """
    T, J, C = positions.shape

    # Random Rotation around Y-axis (vertical)
    theta = np.random.uniform(-0.3, 0.3)  # +/- ~17 degrees
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    # Random Scaling (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)

    # Apply transformations
    # Reshape to (T*J, 3) for matrix multiplication
    flat_pos = positions.reshape(-1, 3)
    aug_pos = np.dot(flat_pos, rotation_matrix) * scale

    return aug_pos.reshape(T, J, C)


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from positions.
    Args:
        positions: (T, J, 3) np.ndarray
    Returns:
        features: (T, InputDim) np.ndarray.
                  Concatenated [Pos, Vel, Acc] flattened over joints.
    """
    # positions shape: (T, J, 3)
    # Velocity: P(t) - P(t-1)
    # Pad first frame with 0
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    # Acceleration: V(t) - V(t-1)
    # Pad first frame with 0
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # Flatten joints and coordinates: (T, J*3)
    T = positions.shape[0]
    pos_flat = positions.reshape(T, -1)
    vel_flat = velocity.reshape(T, -1)
    acc_flat = acceleration.reshape(T, -1)

    # Concatenate based on config
    features_list = [pos_flat]
    if config.USE_VELOCITY:
        features_list.append(vel_flat)
    if config.USE_ACCELERATION:
        features_list.append(acc_flat)

    return np.concatenate(features_list, axis=1)


# ==========================================
# 3. Audio Feature Extraction
# ==========================================


def extract_audio_features(audio_path, num_video_frames):
    """
    Extracts MFCC features from audio and aligns them to video frames.
    Args:
        audio_path (str): Path to .wav file
        num_video_frames (int): Target number of frames for alignment
    Returns:
        mfcc_aligned: (num_video_frames, n_mfcc) np.ndarray
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception:
        # Return zeros if audio load fails
        return np.zeros((num_video_frames, config.NUM_MFCC), dtype=np.float32)

    # Compute MFCC
    # We use standard parameters suitable for speech/audio
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=config.NUM_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )

    mfcc = mfcc_transform(waveform)  # Shape: (1, n_mfcc, time)
    mfcc = mfcc.squeeze(0).transpose(0, 1)  # Shape: (time, n_mfcc)
    mfcc = mfcc.detach().numpy()

    # Align to Video Frames using linear interpolation
    # Current time axis
    curr_time = np.linspace(0, 1, mfcc.shape[0])
    # Target time axis
    target_time = np.linspace(0, 1, num_video_frames)

    mfcc_aligned = np.zeros((num_video_frames, config.NUM_MFCC), dtype=np.float32)

    for i in range(config.NUM_MFCC):
        if mfcc.shape[0] > 1:
            f = scipy.interpolate.interp1d(
                curr_time, mfcc[:, i], kind="linear", fill_value="extrapolate"
            )
            mfcc_aligned[:, i] = f(target_time)
        else:
            mfcc_aligned[:, i] = mfcc[:, i]

    return mfcc_aligned


# ==========================================
# 4. Submission Formatting
# ==========================================


def rle_encode(predictions):
    """
    Run-Length Encoding for predictions.
    Collapses consecutive duplicates and removes background class (0).

    Args:
        predictions (list or np.array): Frame-wise class IDs.

    Returns:
        list: Ordered list of gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = [predictions[0]]
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i - 1]:
            collapsed.append(predictions[i])

    # Remove background class (0)
    final_gestures = [int(p) for p in collapsed if p != config.BACKGROUND_CLASS_ID]

    return final_gestures
