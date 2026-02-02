import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import Config

# Set fixed seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def load_mat_file(path):
    """
    Safely load .mat file using scipy.
    """
    try:
        # struct_as_record=False and squeeze_me=True allow accessing fields as attributes
        return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_skeleton_features(mat_path):
    """
    Extracts and computes skeleton features from the .mat file.

    Returns:
        static_features (np.ndarray): Shape (T, 60) -> Normalized positions (x,y,z) for 20 joints.
        dynamic_features (np.ndarray): Shape (T, 120) -> Velocity (60) + Acceleration (60).
        num_frames (int): Number of frames.
    """
    mat = load_mat_file(mat_path)
    if mat is None or "Video" not in mat:
        return None, None, 0

    video = mat["Video"]
    if not hasattr(video, "Frames"):
        return None, None, 0

    frames = video.Frames
    num_frames = len(frames)

    # 20 joints, 3 coordinates (X, Y, Z)
    # Joints order is assumed to follow the list in description (HipCenter at index 0)
    raw_positions = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    try:
        for i in range(num_frames):
            skeleton = frames[i].Skeleton
            # Handle case where Skeleton might be an array of joints or a struct
            # Based on description, Skeleton contains 'JointsType' and 'WorldPosition'
            # We iterate assuming standard Kinect format where Skeleton is a struct array of 20 joints
            if isinstance(skeleton, np.ndarray) and len(skeleton) == Config.NUM_JOINTS:
                for j in range(Config.NUM_JOINTS):
                    wp = skeleton[j].WorldPosition
                    raw_positions[i, j, 0] = wp.X
                    raw_positions[i, j, 1] = wp.Y
                    raw_positions[i, j, 2] = wp.Z
            else:
                # Fallback: try to access directly if it's a single object (unlikely for 20 joints)
                # or if the structure is different.
                # For robustness, if we fail to parse, we leave as zeros (or could raise error)
                pass
    except Exception as e:
        print(f"Error parsing skeleton in {mat_path}: {e}")
        return None, None, 0

    # 1. Normalization: Subtract Root Joint (HipCenter, index 0) from all joints
    # Shape: (T, 20, 3)
    root_positions = raw_positions[:, 0:1, :]  # Keep dims
    norm_positions = raw_positions - root_positions

    # Flatten spatial dims: (T, 60)
    static_features = norm_positions.reshape(num_frames, -1)

    # 2. Derivatives
    # Velocity: P_t - P_{t-1}
    # Pad first frame with 0
    velocity = np.zeros_like(static_features)
    velocity[1:] = static_features[1:] - static_features[:-1]

    # Acceleration: V_t - V_{t-1}
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # Concatenate for Dynamic Stream: (T, 120)
    dynamic_features = np.concatenate([velocity, acceleration], axis=1)

    return static_features, dynamic_features, num_frames


def extract_audio_features(audio_path, target_num_frames):
    """
    Extracts MFCC features and aligns them to the video frame count.

    Returns:
        mfcc_features (np.ndarray): Shape (T, n_mfcc)
    """
    if not os.path.exists(audio_path):
        # Return zeros if audio missing
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != Config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=Config.AUDIO_SR
            )
            waveform = resampler(waveform)

        # Compute MFCC
        # n_mfcc=13, sample_rate=16000
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SR,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # Shape: (Channels, n_mfcc, time)

        # Average over audio channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0)
        else:
            mfcc = mfcc.squeeze(0)

        # Shape is now (n_mfcc, time_steps)
        # We need to interpolate to match target_num_frames (video frames)
        mfcc = mfcc.unsqueeze(0)  # Add batch dim for interpolate: (1, n_mfcc, time)

        # Interpolate
        mfcc_aligned = torch.nn.functional.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Remove batch dim and transpose to (T, n_mfcc)
        mfcc_aligned = mfcc_aligned.squeeze(0).transpose(0, 1)

        return mfcc_aligned.numpy()

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def create_label_array(labels_json, num_frames):
    """
    Converts JSON label list to a frame-wise label array.
    """
    labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

    if not labels_json:
        return labels

    for entry in labels_json:
        # Metadata uses 1-based indexing from Matlab?
        # Usually 'Begin' and 'End' in the provided mat are frame indices.
        # Assuming 1-based inclusive from Matlab, convert to 0-based Python.
        start = max(0, entry["begin"] - 1)
        end = min(
            num_frames, entry["end"]
        )  # Python slice excludes end, so this works if 'end' is inclusive
        label_id = entry["id"]

        if start < end:
            labels[start:end] = label_id

    return labels


def process_dataset(csv_path, cache_name, load_cached_data=True):
    """
    Loads raw data, processes features, and caches them.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_list = []
            num_samples = int(loaded["num_samples"])
            for i in range(num_samples):
                sample = {
                    "sample_id": str(loaded[f"sample_id_{i}"]),
                    "static": loaded[f"static_{i}"],
                    "dynamic": loaded[f"dynamic_{i}"],
                    "labels": loaded[f"labels_{i}"],
                }
                data_list.append(sample)
            return data_list
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse JSON labels
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    data_list = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # Extract Skeleton
        static_skel, dynamic_skel, num_frames = extract_skeleton_features(mat_path)

        if static_skel is None:
            print(f"Skipping {sample_id} due to skeleton extraction failure.")
            continue

        # Extract Audio
        mfcc = extract_audio_features(audio_path, num_frames)

        # Combine Static Stream: Skeleton (60) + Audio (13) = 73
        static_features = np.concatenate([static_skel, mfcc], axis=1)

        # Labels
        labels = create_label_array(row["parsed_labels"], num_frames)

        data_list.append(
            {
                "sample_id": sample_id,
                "static": static_features.astype(np.float32),
                "dynamic": dynamic_skel.astype(np.float32),
                "labels": labels,
            }
        )

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    save_dict = {"num_samples": len(data_list)}
    for i, item in enumerate(data_list):
        save_dict[f"sample_id_{i}"] = item["sample_id"]
        save_dict[f"static_{i}"] = item["static"]
        save_dict[f"dynamic_{i}"] = item["dynamic"]
        save_dict[f"labels_{i}"] = item["labels"]

    np.savez(cache_path, **save_dict)

    return data_list


class GestureDataset(Dataset):
    def __init__(self, data_list, is_training=True):
        """
        Args:
            data_list (list): List of dictionaries containing features and labels.
            is_training (bool): If True, applies sliding window slicing.
                                If False, returns full sequences.
        """
        self.is_training = is_training
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

        if self.is_training:
            self.samples = self._create_windows(data_list)
        else:
            self.samples = data_list

    def _create_windows(self, data):
        windows = []
        for sample in data:
            static = sample["static"]
            dynamic = sample["dynamic"]
            labels = sample["labels"]

            num_frames = static.shape[0]

            # If sequence is shorter than window, pad it (repeat or zero pad)
            # Here we skip very short sequences or pad if necessary.
            # Given dataset stats (min ~40 frames), we might need padding.
            if num_frames < self.window_size:
                # Pad with zeros
                pad_len = self.window_size - num_frames
                static = np.pad(static, ((0, pad_len), (0, 0)), mode="constant")
                dynamic = np.pad(dynamic, ((0, pad_len), (0, 0)), mode="constant")
                labels = np.pad(
                    labels,
                    (0, pad_len),
                    mode="constant",
                    constant_values=Config.BACKGROUND_CLASS_ID,
                )
                windows.append({"static": static, "dynamic": dynamic, "labels": labels})
                continue

            # Sliding Window
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                end = start + self.window_size
                windows.append(
                    {
                        "static": static[start:end],
                        "dynamic": dynamic[start:end],
                        "labels": labels[start:end],
                    }
                )

        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Convert to Torch Tensors
        # Features: (T, F)
        static_tensor = torch.FloatTensor(item["static"])
        dynamic_tensor = torch.FloatTensor(item["dynamic"])

        # Early Fusion: Concatenate all features (Cite solution_lesson_node_00021)
        # Shape: (T, 73 + 120) = (T, 193)
        features = torch.cat([static_tensor, dynamic_tensor], dim=1)

        if self.is_training:
            labels_tensor = torch.LongTensor(item["labels"])
            return features, labels_tensor
        else:
            # For inference, we might need the sample ID
            return features, item["sample_id"]


def get_datasets(load_cached_data=True):
    """
    Main entry point to get dataset objects.
    """
    Config.setup_directories()

    # Load Data Lists
    train_data = process_dataset(Config.TRAIN_CSV, "train_features", load_cached_data)
    val_data = process_dataset(Config.VAL_CSV, "val_features", load_cached_data)
    test_data = process_dataset(Config.TEST_CSV, "test_features", load_cached_data)

    # Create Datasets
    train_dataset = GestureDataset(train_data, is_training=True)
    val_dataset = GestureDataset(
        val_data, is_training=True
    )  # Validation also windowed for loss calc?
    # Usually validation is done on full sequences for metric calculation,
    # but for monitoring loss during training, windowing is fine.
    # However, to compute Levenshtein, we need full sequences.
    # Let's keep validation as full sequences to allow metric computation.
    val_dataset_full = GestureDataset(val_data, is_training=False)

    test_dataset = GestureDataset(test_data, is_training=False)

    return train_dataset, val_dataset_full, test_dataset
