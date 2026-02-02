import os
import json
import numpy as np
import pandas as pd
import scipy.io
import librosa
import warnings
from library.config import Config

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class PolymorphicMatLoader:
    """
    Robustly loads .mat files handling various structural inconsistencies
    in the Skeleton field (struct array vs cell array vs primitive).
    """

    # Mapping from JointsType string to index (0-19)
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

    @staticmethod
    def load_skeleton_frame(frame_data):
        """
        Parses a single frame's skeleton data into a (20, 3) numpy array.
        Returns zeros if no valid skeleton is found.
        """
        joints_pos = np.zeros((20, 3), dtype=np.float32)

        if not hasattr(frame_data, "Skeleton"):
            return joints_pos

        skeleton = frame_data.Skeleton

        # Handle case where Skeleton is empty or 0
        if isinstance(skeleton, (int, float)) or skeleton is None:
            return joints_pos
        if isinstance(skeleton, np.ndarray) and skeleton.size == 0:
            return joints_pos

        # Normalize to a list of joint objects
        joints_list = []

        # Case 1: Skeleton is a single object (maybe containing an array?)
        if not isinstance(skeleton, (list, np.ndarray)):
            # If it's a single struct, check if it's a joint or a user
            if hasattr(skeleton, "JointsType"):
                joints_list = [skeleton]
            else:
                # Might be a user struct containing joints?
                # For this dataset, usually Skeleton is the array of joints directly
                pass
        # Case 2: Skeleton is an array/list
        else:
            # Check if it's a flat array of joints
            if len(skeleton) > 0:
                # If the first element has JointsType, assume it's the list of joints
                if hasattr(skeleton[0], "JointsType"):
                    joints_list = skeleton
                # If nested (e.g. multiple users), take the first user's joints if available
                elif hasattr(skeleton[0], "Skeleton"):
                    # Recursive check not implemented for depth > 1, assuming single user primary
                    pass

        # Extract positions
        for joint in joints_list:
            if hasattr(joint, "JointsType") and hasattr(joint, "WorldPosition"):
                j_type = str(joint.JointsType)
                pos = joint.WorldPosition

                if j_type in PolymorphicMatLoader.JOINT_MAP:
                    idx = PolymorphicMatLoader.JOINT_MAP[j_type]
                    # Check if pos has X, Y, Z fields
                    if hasattr(pos, "X") and hasattr(pos, "Y") and hasattr(pos, "Z"):
                        joints_pos[idx] = [float(pos.X), float(pos.Y), float(pos.Z)]

        return joints_pos

    @staticmethod
    def load_video_data(mat_path):
        """
        Loads the .mat file and extracts the full skeleton sequence.
        Returns:
            skeleton_data: (T, 20, 3) in meters
            frame_rate: float
        """
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            # print(f"Error loading {mat_path}: {e}")
            return None, 30.0  # Default fallback

        if "Video" not in mat:
            return None, 30.0

        video = mat["Video"]

        # Get Frame Rate
        frame_rate = 30.0
        if hasattr(video, "FrameRate"):
            frame_rate = float(video.FrameRate)

        # Get Frames
        if not hasattr(video, "Frames"):
            return None, frame_rate

        frames = video.Frames

        # Handle case where Frames is a single object or list
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        num_frames = len(frames)
        skeleton_seq = np.zeros((num_frames, 20, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            skeleton_seq[i] = PolymorphicMatLoader.load_skeleton_frame(frame)

        # Convert mm to meters
        skeleton_seq = skeleton_seq / 1000.0

        return skeleton_seq, frame_rate


def compute_kinematics(positions):
    """
    Computes Velocity and Acceleration from positions.
    Input: (T, J, 3)
    Output: (T, J, 9) -> [Pos, Vel, Acc]
    """
    # positions: T x J x 3
    # Velocity: V[t] = P[t] - P[t-1]
    # Acceleration: A[t] = V[t] - V[t-1]

    # Pad first frame for velocity
    padded_pos = np.pad(positions, ((1, 0), (0, 0), (0, 0)), mode="edge")
    velocity = padded_pos[1:] - padded_pos[:-1]  # T x J x 3

    # Pad first frame for acceleration
    padded_vel = np.pad(velocity, ((1, 0), (0, 0), (0, 0)), mode="edge")
    acceleration = padded_vel[1:] - padded_vel[:-1]  # T x J x 3

    # Concatenate features
    # Shape: (T, J, 9)
    return np.concatenate([positions, velocity, acceleration], axis=-1)


def extract_audio_features(audio_path, target_num_frames, video_fps):
    """
    Extracts MFCCs from audio and aligns them to the video frame count.
    Returns: (T, 13)
    """
    target_sr = 16000
    n_mfcc = Config.AUDIO_FEATURES

    # Initialize empty if file doesn't exist
    if not os.path.exists(audio_path):
        return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
    except Exception:
        return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

    if len(y) == 0:
        return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

    # Calculate hop length to match video FPS approximately
    # hop_length = samples / frame
    hop_length = int(sr / video_fps)
    if hop_length < 1:
        hop_length = 1

    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # mfccs shape: (n_mfcc, T_audio)
    mfccs = mfccs.T  # (T_audio, n_mfcc)

    # Align to video frames
    curr_frames = mfccs.shape[0]

    if curr_frames == 0:
        return np.zeros((target_num_frames, n_mfcc), dtype=np.float32)

    if curr_frames != target_num_frames:
        # Resample along time axis
        # Create interpolation indices
        x_old = np.linspace(0, 1, curr_frames)
        x_new = np.linspace(0, 1, target_num_frames)

        new_mfccs = np.zeros((target_num_frames, n_mfcc), dtype=np.float32)
        for i in range(n_mfcc):
            new_mfccs[:, i] = np.interp(x_new, x_old, mfccs[:, i])
        return new_mfccs

    return mfccs.astype(np.float32)


def levenshtein_score(preds, targets):
    """
    Computes the Levenshtein distance metric.
    preds: list of lists of predicted gesture IDs
    targets: list of lists of ground truth gesture IDs
    Returns: scalar score (Distance / Total Truth Length)
    """
    total_dist = 0
    total_len = 0

    for p_seq, t_seq in zip(preds, targets):
        # Calculate Levenshtein distance between two sequences
        n = len(p_seq)
        m = len(t_seq)

        if n == 0:
            dist = m
        elif m == 0:
            dist = n
        else:
            # DP Matrix
            d = np.zeros((n + 1, m + 1), dtype=int)
            for i in range(n + 1):
                d[i, 0] = i
            for j in range(m + 1):
                d[0, j] = j

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = 0 if p_seq[i - 1] == t_seq[j - 1] else 1
                    d[i, j] = min(
                        d[i - 1, j] + 1,  # deletion
                        d[i, j - 1] + 1,  # insertion
                        d[i - 1, j - 1] + cost,
                    )  # substitution
            dist = d[n, m]

        total_dist += dist
        total_len += m

    if total_len == 0:
        return 0.0

    return total_dist / total_len


def load_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads the dataset, processing raw files if cache is not available.
    Returns a dictionary containing flattened arrays and indices.

    Structure of returned dict (reconstructed from flattened):
    - skeletons: list of (T, 20, 3) arrays
    - audio: list of (T, 13) arrays
    - labels: list of (T,) arrays
    - sample_ids: list of strings
    - gt_sequences: list of lists of ints
    """

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)

            # Reconstruct lists from flattened arrays
            all_skeletons = data["all_skeletons"]
            all_audio = data["all_audio"]
            all_labels = data["all_labels"]
            indices = data["indices"]  # (N, 2) start, end
            sample_ids = data["sample_ids"]
            gt_seq_strs = data["gt_seq_strs"]

            skeletons = []
            audio = []
            labels = []
            gt_sequences = []

            for i in range(len(indices)):
                start, end = indices[i]
                skeletons.append(all_skeletons[start:end])
                audio.append(all_audio[start:end])
                labels.append(all_labels[start:end])

                # Parse GT sequence string "1 2 3" -> [1, 2, 3]
                s = str(gt_seq_strs[i])
                if len(s) > 0:
                    gt_sequences.append([int(x) for x in s.split()])
                else:
                    gt_sequences.append([])

            return {
                "skeletons": skeletons,
                "audio": audio,
                "labels": labels,
                "sample_ids": sample_ids.tolist(),
                "gt_sequences": gt_sequences,
            }
        except Exception as e:
            # print(f"Cache load failed: {e}. Reprocessing...")
            pass

    # 2. Process from scratch
    # print(f"Processing dataset from {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Containers for flattened data
    flat_skeletons = []
    flat_audio = []
    flat_labels = []
    indices = []
    sample_ids = []
    gt_seq_strs = []

    current_idx = 0

    for _, row in df.iterrows():
        sid = row["sample_id"]
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Load Skeleton
        skel, fps = PolymorphicMatLoader.load_video_data(mat_path)

        if skel is None:
            # Skip corrupted samples or handle gracefully
            # print(f"Warning: Failed to load skeleton for {sid}")
            continue

        num_frames = skel.shape[0]

        # Load Audio
        aud = extract_audio_features(audio_path, num_frames, fps)

        # Create Frame-wise Labels
        lbl_array = np.zeros(num_frames, dtype=np.int32)
        gt_seq = []

        # Parse labels JSON
        if isinstance(row["labels"], str):
            try:
                labels_list = json.loads(row["labels"])
                for l in labels_list:
                    gid = int(l["id"])
                    start = int(l["begin"]) - 1  # 1-based to 0-based
                    end = int(l["end"])  # inclusive in matlab usually implies range

                    # Clip to valid range
                    start = max(0, start)
                    end = min(num_frames, end)

                    if end > start:
                        lbl_array[start:end] = gid
                        gt_seq.append(gid)
            except:
                pass

        # Append to lists
        flat_skeletons.append(skel)
        flat_audio.append(aud)
        flat_labels.append(lbl_array)

        indices.append([current_idx, current_idx + num_frames])
        current_idx += num_frames

        sample_ids.append(sid)
        gt_seq_strs.append(" ".join(map(str, gt_seq)))

    # Concatenate
    if len(flat_skeletons) > 0:
        all_skeletons = np.concatenate(flat_skeletons, axis=0)
        all_audio = np.concatenate(flat_audio, axis=0)
        all_labels = np.concatenate(flat_labels, axis=0)
    else:
        all_skeletons = np.zeros((0, 20, 3), dtype=np.float32)
        all_audio = np.zeros((0, 13), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int32)

    indices = np.array(indices, dtype=np.int32)
    sample_ids = np.array(sample_ids)
    gt_seq_strs = np.array(gt_seq_strs)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        all_skeletons=all_skeletons,
        all_audio=all_audio,
        all_labels=all_labels,
        indices=indices,
        sample_ids=sample_ids,
        gt_seq_strs=gt_seq_strs,
    )

    # Return structured format
    # Re-construct lists (same logic as loading)
    # We can just return the lists we built before flattening to save time
    # But for consistency with the cache-load path, we return the lists of arrays.

    return {
        "skeletons": flat_skeletons,
        "audio": flat_audio,
        "labels": flat_labels,
        "sample_ids": sample_ids.tolist(),
        "gt_sequences": [list(map(int, s.split())) if s else [] for s in gt_seq_strs],
    }
