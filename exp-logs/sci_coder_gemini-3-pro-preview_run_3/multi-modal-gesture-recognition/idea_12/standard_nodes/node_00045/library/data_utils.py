import os
import json
import scipy.io
import soundfile as sf
import pandas as pd
import numpy as np
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================

# Mapping from JointType string to index (0-19)
JOINT_MAP = {
    "hipcenter": 0,
    "spine": 1,
    "shouldercenter": 2,
    "head": 3,
    "shoulderleft": 4,
    "elbowleft": 5,
    "wristleft": 6,
    "handleft": 7,
    "shoulderright": 8,
    "elbowright": 9,
    "wristright": 10,
    "handright": 11,
    "hipleft": 12,
    "kneeleft": 13,
    "ankleleft": 14,
    "footleft": 15,
    "hipright": 16,
    "kneeright": 17,
    "ankleright": 18,
    "footright": 19,
}

# ==========================================
# File I/O Utilities
# ==========================================


def load_mat_file(path):
    """
    Safely load a .mat file using scipy.io.
    Returns the mat dictionary or None if failed.
    """
    try:
        # struct_as_record=False loads structs as objects
        # squeeze_me=True simplifies 1x1 arrays
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading MAT file {path}: {e}")
        return None


def load_audio(path):
    """
    Load an audio file.
    Returns (data, samplerate).
    """
    try:
        return sf.read(path)
    except Exception as e:
        print(f"Error loading audio {path}: {e}")
        return None, None


def parse_labels(labels_json_str):
    """
    Parse the JSON string of labels from the metadata CSV.
    Returns a list of dictionaries.
    """
    try:
        if isinstance(labels_json_str, str):
            return json.loads(labels_json_str)
        return []
    except Exception:
        return []


def save_submission(predictions, sample_ids, output_path):
    """
    Save predictions to a CSV file in the format required for submission.

    Args:
        predictions: List of lists/arrays containing gesture IDs.
        sample_ids: List of sequence IDs (e.g., 'Sample00001').
        output_path: Path to save the CSV.
    """
    rows = []
    for sid, preds in zip(sample_ids, predictions):
        # Format: SessionID,label1,label2,...
        # Remove duplicates if strictly required, but RLE is usually handled before this.
        # Ensure preds are integers
        pred_str = ",".join(map(str, preds))
        rows.append(f"{sid},{pred_str}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")

    print(f"Submission saved to {output_path}")


# ==========================================
# Skeleton Data Parser
# ==========================================


def load_skeleton_data(path):
    """
    Parses the specific Skeleton structure from the challenge .mat file.

    Structure expectation:
    mat.Video.Frames is an array of frame objects.
    Each frame object has a 'Skeleton' field.
    'Skeleton' is an array of joint objects (one per joint).
    Each joint object has 'JointsType' and 'WorldPosition' (X, Y, Z).

    Returns:
        numpy.ndarray: Shape (NumFrames, 20, 3) containing X,Y,Z coordinates.
                       Returns None if parsing fails.
    """
    mat = load_mat_file(path)
    if mat is None:
        return None

    try:
        if not hasattr(mat, "Video"):
            return None
        video = mat.Video

        if not hasattr(video, "Frames"):
            return None
        frames = video.Frames

        # Handle case where Frames is a single object (1 frame video)
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        num_frames = len(frames)
        num_joints = 20

        # Initialize tensor (T, J, 3)
        skeleton_tensor = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

        for f_idx, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel_data = frame.Skeleton

            # Check if Skeleton is valid
            if isinstance(skel_data, (list, np.ndarray)):
                # Iterate over joints in the skeleton array
                for joint in skel_data:
                    # Check for JointsType
                    j_type = None
                    if hasattr(joint, "JointsType"):
                        j_type = str(joint.JointsType).lower()

                    if j_type in JOINT_MAP:
                        j_idx = JOINT_MAP[j_type]

                        # Extract WorldPosition
                        if hasattr(joint, "WorldPosition"):
                            wp = joint.WorldPosition
                            # Check if wp has X, Y, Z attributes
                            if (
                                hasattr(wp, "X")
                                and hasattr(wp, "Y")
                                and hasattr(wp, "Z")
                            ):
                                skeleton_tensor[f_idx, j_idx, 0] = float(wp.X)
                                skeleton_tensor[f_idx, j_idx, 1] = float(wp.Y)
                                skeleton_tensor[f_idx, j_idx, 2] = float(wp.Z)

            # Fallback: sometimes single struct if only 1 joint? Unlikely for full body.

        return skeleton_tensor

    except Exception as e:
        # In production/debugging, un-comment the next line to see specific parsing errors
        # print(f"Error parsing skeleton in {path}: {e}")
        return None


# ==========================================
# Dataset Caching & Loading
# ==========================================


def get_dataset_metadata(split):
    """
    Load the metadata DataFrame for a specific split ('train', 'val', 'test').
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def load_cached_dataset(split, load_cached_data=True):
    """
    Loads the dataset for a given split.
    Implements deterministic caching using .npz files.

    Strategy:
    - Flatten all sequences into a single large array to avoid pickle.
    - Store offsets/lengths to reconstruct individual sequences.
    - Store labels and sample IDs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: {
            'features': List[np.ndarray], # List of (T, 20, 3) arrays
            'labels': List[List[dict]],   # List of label lists (for train/val)
            'sample_ids': List[str]       # List of sample IDs
        }
    """
    Config.setup()  # Ensure directories exist
    cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split}.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} dataset from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=False)  # Strictly no pickle

            flat_features = data["flat_features"]
            lengths = data["lengths"]
            sample_ids = data["sample_ids"]

            # Reconstruct features list
            features_list = []
            current_idx = 0
            for length in lengths:
                features_list.append(flat_features[current_idx : current_idx + length])
                current_idx += length

            # Reconstruct labels (stored as JSON strings in a numpy array)
            labels_json = data["labels_json"]
            labels_list = [json.loads(l) for l in labels_json]

            return {
                "features": features_list,
                "labels": labels_list,
                "sample_ids": sample_ids.tolist(),
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {split} dataset from raw files...")
    df = get_dataset_metadata(split)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLES)
        print(f"DEBUG MODE: Processing only {len(df)} samples.")

    all_features = []
    all_lengths = []
    all_labels_json = []
    all_sample_ids = []

    for _, row in df.iterrows():
        # Load Skeleton
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        skel_data = load_skeleton_data(mat_path)

        if skel_data is None:
            # Fallback for missing/corrupt data: create empty sequence of length 1
            # to maintain alignment
            print(f"Warning: Could not load skeleton for {row['sample_id']}")
            skel_data = np.zeros((1, 20, 3), dtype=np.float32)

        all_features.append(skel_data)
        all_lengths.append(skel_data.shape[0])

        # Labels
        # In metadata CSV, labels are already JSON strings. We keep them as strings for storage.
        all_labels_json.append(row["labels"])
        all_sample_ids.append(str(row["sample_id"]))

    # 3. Save to Cache
    # Flatten features
    flat_features = np.concatenate(all_features, axis=0).astype(np.float32)
    lengths_arr = np.array(all_lengths, dtype=np.int32)
    labels_arr = np.array(all_labels_json)  # Array of strings
    ids_arr = np.array(all_sample_ids)  # Array of strings

    np.savez(
        cache_path,
        flat_features=flat_features,
        lengths=lengths_arr,
        labels_json=labels_arr,
        sample_ids=ids_arr,
    )
    print(f"Saved {split} dataset to cache: {cache_path}")

    # Return in the structured format
    # Parse labels for immediate use
    parsed_labels = [json.loads(l) for l in all_labels_json]

    return {
        "features": all_features,
        "labels": parsed_labels,
        "sample_ids": all_sample_ids,
    }
