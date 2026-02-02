import os
import json
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import pandas as pd
import nltk
import soundfile as sf
import torchaudio
from library.config import Config

# ==========================================
# 1. Polymorphic Data Loading
# ==========================================


def load_mat_polymorphic(mat_path):
    """
    Robustly loads a .mat file and extracts the skeleton world positions.
    Handles variations in the internal structure of the .mat files (struct arrays vs objects).

    Args:
        mat_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (T, 20, 3). Returns None if extraction fails.
    """
    try:
        # Load with squeeze_me=True to simplify arrays, struct_as_record=False to get objects
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

    if "Video" not in mat:
        return None

    video = mat["Video"]
    # Unwrap 0-d array if necessary (squeeze_me=True effect)
    if isinstance(video, np.ndarray) and video.ndim == 0:
        video = video.item()

    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames

    # Handle case where Frames is a single object (1 frame) or an array
    if not isinstance(frames, (list, np.ndarray)):
        frames = [frames]

    num_frames = len(frames)
    num_joints = 20
    # Pre-allocate: T x Joints x 3 (X, Y, Z)
    skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    # Ordered list of joints as per dataset description
    joint_names = [
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

    for t, frame in enumerate(frames):
        # Check if Skeleton exists
        if not hasattr(frame, "Skeleton"):
            continue

        skel = frame.Skeleton

        # If multiple skeletons are tracked, skel might be an array.
        # We take the first one (UserIndex usually aligns with this, but for simplicity we take index 0)
        if isinstance(skel, (list, np.ndarray)):
            if len(skel) == 0:
                continue
            skel = skel[0]

        # Extract joints
        # Case 1: Skeleton has fields for each joint (e.g. skel.Head)
        # Case 2: Skeleton has a 'Joints' array?
        # Based on description: "Skeleton... JointsType... WorldPosition"
        # It implies Skeleton might be an array of structures if accessed differently,
        # but usually in these datasets, the fields are direct properties if struct_as_record=False.

        # We try to access attributes by name
        try:
            for j_idx, j_name in enumerate(joint_names):
                if hasattr(skel, j_name):
                    joint_node = getattr(skel, j_name)
                    if hasattr(joint_node, "WorldPosition"):
                        pos = joint_node.WorldPosition
                        # pos might be an object with X,Y,Z or an array
                        if (
                            hasattr(pos, "X")
                            and hasattr(pos, "Y")
                            and hasattr(pos, "Z")
                        ):
                            skeleton_data[t, j_idx, 0] = pos.X
                            skeleton_data[t, j_idx, 1] = pos.Y
                            skeleton_data[t, j_idx, 2] = pos.Z
                        elif isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                            skeleton_data[t, j_idx, :] = pos[:3]
        except Exception:
            # Fallback or skip frame if structure is completely unexpected
            pass

    return skeleton_data


# ==========================================
# 2. Metric Calculation
# ==========================================


def levenshtein_distance(preds, targets):
    """
    Computes the Levenshtein distance between prediction and target sequences.

    Args:
        preds (list): List of predicted gesture IDs.
        targets (list): List of ground truth gesture IDs.

    Returns:
        int: Edit distance.
    """
    # Use NLTK's implementation which is standard and efficient
    return nltk.edit_distance(preds, targets)


# ==========================================
# 3. Run-Length Encoding / Decoding
# ==========================================


def rle_encode(frame_preds):
    """
    Compresses frame-wise predictions into a sequence of gesture IDs.
    Merges consecutive identical labels and removes background (class 0).

    Args:
        frame_preds (list or np.ndarray): Sequence of class IDs.

    Returns:
        list: List of gesture IDs (excluding background).
    """
    if len(frame_preds) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = [frame_preds[0]]
    for i in range(1, len(frame_preds)):
        if frame_preds[i] != frame_preds[i - 1]:
            collapsed.append(frame_preds[i])

    # Remove background (0)
    final_sequence = [x for x in collapsed if x != 0]
    return final_sequence


def filter_short_segments(frame_preds, min_duration=5):
    """
    Post-processing: Removes segments that are shorter than min_duration.

    Args:
        frame_preds (np.ndarray): Frame-wise predictions.
        min_duration (int): Minimum duration in frames.

    Returns:
        np.ndarray: Filtered frame-wise predictions.
    """
    if len(frame_preds) == 0:
        return frame_preds

    # Find segments: (label, start, end)
    segments = []
    if len(frame_preds) > 0:
        curr_label = frame_preds[0]
        curr_start = 0
        for i in range(1, len(frame_preds)):
            if frame_preds[i] != curr_label:
                segments.append({"label": curr_label, "start": curr_start, "end": i})
                curr_label = frame_preds[i]
                curr_start = i
        segments.append(
            {"label": curr_label, "start": curr_start, "end": len(frame_preds)}
        )

    # Filter
    filtered_preds = np.array(frame_preds, copy=True)

    # We iterate and if a segment is too short, we can either:
    # 1. Replace it with the previous label (simple smoothing)
    # 2. Replace it with background (0)
    # The prompt suggests "Min Duration Filter: Explicitly remove segments".
    # We will replace short segments with the label of the longer neighbor or background.
    # For simplicity and robustness, we replace with the previous valid segment's label or 0.

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        if duration < min_duration:
            # Determine replacement label
            # Heuristic: replace with previous label if it exists, else next, else 0
            replacement = 0
            if i > 0:
                replacement = segments[i - 1]["label"]
            elif i < len(segments) - 1:
                replacement = segments[i + 1]["label"]

            filtered_preds[seg["start"] : seg["end"]] = replacement

            # Update segment list for subsequent logic?
            # Ideally we would re-run RLE, but single pass is usually sufficient for small glitches.

    return filtered_preds


# ==========================================
# 4. Loss Function
# ==========================================


class TruncatedMSELoss(nn.Module):
    """
    Log-Space Smoothing Loss.
    Penalizes rapid changes in log-probabilities to enforce temporal smoothness.
    L = mean( min( (log_p_t - log_p_{t-1})^2, threshold^2 ) )
    """

    def __init__(self, threshold=1.0):
        super(TruncatedMSELoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, log_probs):
        """
        Args:
            log_probs: (Batch, Time, Classes) - Log probabilities from the model.
        """
        # Calculate temporal difference: log_P_t - log_P_{t-1}
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        diff_sq = diff**2

        # Truncate
        loss = torch.clamp(diff_sq, max=self.threshold_sq)

        return loss.mean()


# ==========================================
# 5. Deterministic Stats Computation (Cached)
# ==========================================


def compute_stats(load_cached_data=True):
    """
    Computes mean and std for Skeleton (Pos, Vel, Acc) and Audio (MFCC).
    Implements caching to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from Config.STATS_FILE.

    Returns:
        dict: {'skeleton_mean': np.array, 'skeleton_std': np.array,
               'audio_mean': np.array, 'audio_std': np.array}
    """
    stats_path = Config.STATS_FILE

    # 1. Try Load
    if load_cached_data and os.path.exists(stats_path):
        print(f"Loading cached stats from {stats_path}")
        data = np.load(stats_path)
        return {
            "skeleton_mean": data["skeleton_mean"],
            "skeleton_std": data["skeleton_std"],
            "audio_mean": data["audio_mean"],
            "audio_std": data["audio_std"],
        }

    print("Computing normalization stats from training data (this may take a while)...")

    # 2. Compute
    df_train = pd.read_csv(Config.TRAIN_METADATA)

    # Accumulators
    # Skeleton: 20 joints * 3 coords * 3 derivatives = 180 channels
    # We compute stats per channel
    skel_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    skel_sq_sum = np.zeros(Config.SKELETON_INPUT_DIM)
    skel_count = 0

    # Audio: 13 MFCCs
    audio_sum = np.zeros(Config.AUDIO_INPUT_DIM)
    audio_sq_sum = np.zeros(Config.AUDIO_INPUT_DIM)
    audio_count = 0

    # Audio Transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=Config.AUDIO_MFCC_N_MFCC
    )

    for idx, row in df_train.iterrows():
        # --- Process Skeleton ---
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        skel_pos = load_mat_polymorphic(mat_path)  # (T, 20, 3)

        if skel_pos is not None and len(skel_pos) > 2:
            # Compute Derivatives
            # Pad first frame for velocity, first two for accel to keep shape
            # Or just use diff and valid frames. We use valid frames for stats.

            # Velocity: P_t - P_{t-1}
            vel = np.diff(skel_pos, axis=0)  # (T-1, 20, 3)
            # Acceleration: V_t - V_{t-1}
            acc = np.diff(vel, axis=0)  # (T-2, 20, 3)

            # Align lengths for concatenation (trim start)
            # P: [2:], V: [1:], A: [:]
            p_trim = skel_pos[2:]
            v_trim = vel[1:]
            a_trim = acc

            # Flatten joints and coords: (T', 20*3)
            T_valid = len(a_trim)
            p_flat = p_trim.reshape(T_valid, -1)
            v_flat = v_trim.reshape(T_valid, -1)
            a_flat = a_trim.reshape(T_valid, -1)

            # Concatenate: (T', 180)
            # Order: P, V, A
            features = np.concatenate([p_flat, v_flat, a_flat], axis=1)

            skel_sum += np.sum(features, axis=0)
            skel_sq_sum += np.sum(features**2, axis=0)
            skel_count += T_valid

        # --- Process Audio ---
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
        if os.path.exists(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)
                # Resample if needed (dataset analysis said 16k, config assumes 16k)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                # Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
                mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()  # (time, n_mfcc)

                audio_sum += np.sum(mfcc, axis=0)
                audio_sq_sum += np.sum(mfcc**2, axis=0)
                audio_count += mfcc.shape[0]
            except Exception:
                pass

    # Finalize Stats
    # Avoid division by zero
    skel_count = max(skel_count, 1)
    audio_count = max(audio_count, 1)

    skel_mean = skel_sum / skel_count
    skel_std = np.sqrt((skel_sq_sum / skel_count) - (skel_mean**2) + 1e-6)

    audio_mean = audio_sum / audio_count
    audio_std = np.sqrt((audio_sq_sum / audio_count) - (audio_mean**2) + 1e-6)

    # 3. Save
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    np.savez(
        stats_path,
        skeleton_mean=skel_mean,
        skeleton_std=skel_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )

    print(f"Stats computed and saved to {stats_path}")

    return {
        "skeleton_mean": skel_mean,
        "skeleton_std": skel_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }
