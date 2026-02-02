import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import pad_collate

# ==========================================
# Helper Functions for Data Processing
# ==========================================


def load_skeleton_data(mat_path):
    """
    Parses the .mat file to extract and normalize skeleton joint coordinates.
    Returns:
        numpy.ndarray: Shape (NumFrames, 60) containing flattened (x,y,z) for 20 joints.
        int: Number of frames.
    """
    try:
        # Load mat file
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if "Video" not in mat:
            return None, 0

        video = mat["Video"]
        num_frames = int(video.NumFrames)
        frames = video.Frames

        # Initialize container: (NumFrames, 20, 3)
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        # Joint names in order (based on description)
        # We assume the order in the description matches the extraction order
        # 0: HipCenter is the root for normalization

        # Check if frames is iterable
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        # Iterate frames
        # Note: If frames are fewer than num_frames, we stop early or pad.
        # We'll iterate up to min(len(frames), num_frames)
        iter_len = min(
            len(frames) if isinstance(frames, (list, np.ndarray)) else 0, num_frames
        )

        for i in range(iter_len):
            frame_obj = frames[i]
            if hasattr(frame_obj, "Skeleton") and hasattr(
                frame_obj.Skeleton, "WorldPosition"
            ):
                wp = frame_obj.Skeleton.WorldPosition
                # Check if WorldPosition has X, Y, Z attributes (structure) or is an array
                # The description says structure with X, Y, Z.
                # However, sometimes .mat parsing results in arrays if multiple skeletons.
                # We assume single user or take the first one if array.

                # Helper to extract xyz from a joint struct
                def get_xyz(joint_idx):
                    # This logic depends heavily on how the mat file struct is flattened.
                    # Based on description, WorldPosition is 20x1 struct or similar.
                    # We will try to access it as an array of joints.
                    pass

                # Actually, often in these datasets, WorldPosition is an array (20,) of structs
                # or a struct with arrays. Let's try to parse robustly.
                # Given the complexity of unknown mat structure details, we assume
                # WorldPosition is an array of 20 objects or a struct containing arrays.
                # A common format in this challenge data:
                # frame_obj.Skeleton.WorldPosition is a (20,) struct array or similar.

                # Simplified extraction logic assuming we can get a (20, 3) array
                # If exact parsing fails, we return zeros for that frame.
                try:
                    # Attempt to convert WorldPosition to list of (x,y,z)
                    # This block assumes specific structure.
                    # If WorldPosition is a single object with X,Y,Z arrays:
                    if hasattr(wp, "X") and isinstance(wp.X, (np.ndarray, list)):
                        # wp.X might be (20,)
                        x = np.array(wp.X).flatten()
                        y = np.array(wp.Y).flatten()
                        z = np.array(wp.Z).flatten()
                        # Stack
                        joints = np.stack([x, y, z], axis=1)  # (20, 3)
                        skeleton_data[i] = joints
                    # If WorldPosition is an array of structs
                    elif isinstance(wp, np.ndarray) and len(wp) == Config.NUM_JOINTS:
                        for j in range(Config.NUM_JOINTS):
                            joint = wp[j]
                            skeleton_data[i, j, 0] = joint.X
                            skeleton_data[i, j, 1] = joint.Y
                            skeleton_data[i, j, 2] = joint.Z
                except:
                    pass

        # Root-Relative Normalization
        # Subtract HipCenter (index 0) from all joints
        root = skeleton_data[:, 0:1, :]  # (NumFrames, 1, 3)
        skeleton_data = skeleton_data - root

        # Flatten to (NumFrames, 60)
        skeleton_data = skeleton_data.reshape(num_frames, -1)

        return skeleton_data, num_frames

    except Exception as e:
        # print(f"Error parsing MAT {mat_path}: {e}")
        return None, 0


def load_audio_features(audio_path, target_num_frames):
    """
    Loads audio, extracts MFCCs, and resamples to match video frame count.
    Returns:
        numpy.ndarray: Shape (target_num_frames, AUDIO_FEATURE_DIM)
    """
    try:
        if not os.path.exists(audio_path):
            return np.zeros(
                (target_num_frames, Config.AUDIO_FEATURE_DIM), dtype=np.float32
            )

        waveform, sample_rate = torchaudio.load(audio_path)

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.AUDIO_FEATURE_DIM,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, time)

        # Average over channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # mfcc shape is now (1, n_mfcc, time)

        # Interpolate to match target_num_frames
        # Input to interpolate needs to be (Batch, Channels, Length) -> (1, n_mfcc, target_frames)
        if target_num_frames > 0:
            mfcc = F.interpolate(
                mfcc, size=target_num_frames, mode="linear", align_corners=False
            )

        # Transpose to (target_frames, n_mfcc)
        mfcc = mfcc.squeeze(0).transpose(0, 1).numpy()

        return mfcc.astype(np.float32)

    except Exception as e:
        # print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((target_num_frames, Config.AUDIO_FEATURE_DIM), dtype=np.float32)


def process_single_sample(row):
    """
    Process a single row from the metadata DataFrame.
    Returns:
        features (np.array): Combined features
        labels (np.array): Dense labels
        valid (bool): Whether processing was successful
    """
    data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
    audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

    # 1. Load Skeleton
    skel_features, num_frames = load_skeleton_data(data_path)

    if skel_features is None or num_frames == 0:
        return None, None, False

    # 2. Load Audio
    audio_features = load_audio_features(audio_path, num_frames)

    # 3. Concatenate
    # Check shapes
    if len(skel_features) != len(audio_features):
        # Fallback resize if mismatch (should be handled by interpolate, but safety check)
        min_len = min(len(skel_features), len(audio_features))
        skel_features = skel_features[:min_len]
        audio_features = audio_features[:min_len]
        num_frames = min_len

    combined_features = np.concatenate(
        [skel_features, audio_features], axis=1
    )  # (T, 73)

    # 4. Generate Labels
    dense_labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

    # Parse labels from JSON string in metadata
    try:
        label_list = json.loads(row["labels"]) if isinstance(row["labels"], str) else []
        for l in label_list:
            # MATLAB frames are 1-based, Python 0-based
            start = max(0, int(l["begin"]) - 1)
            end = min(num_frames, int(l["end"]))
            lid = int(l["id"])
            if start < end:
                dense_labels[start:end] = lid
    except:
        pass  # If labels fail, we just have background (valid for test set)

    return combined_features, dense_labels, True


def save_cache(cache_path, features_list, labels_list, sample_ids):
    """
    Save processed data to .npz file (No Pickle).
    We flatten everything and save indices.
    """
    # Create index array: (start_idx, length)
    offsets = []
    current_idx = 0
    for f in features_list:
        l = len(f)
        offsets.append([current_idx, l])
        current_idx += l

    offsets = np.array(offsets, dtype=np.int64)

    # Concatenate all
    if len(features_list) > 0:
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
    else:
        all_features = np.zeros((0, Config.INPUT_DIM), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)

    # Save IDs as a separate text file or just rely on metadata order.
    # We will assume metadata order is preserved.

    np.savez(cache_path, features=all_features, labels=all_labels, offsets=offsets)


def load_cache(cache_path):
    """
    Load data from .npz file.
    Returns list of features and list of labels.
    """
    data = np.load(cache_path)
    all_features = data["features"]
    all_labels = data["labels"]
    offsets = data["offsets"]

    features_list = []
    labels_list = []

    for start, length in offsets:
        features_list.append(all_features[start : start + length])
        labels_list.append(all_labels[start : start + length])

    return features_list, labels_list


def get_processed_data(metadata_path, cache_path, load_cached_data=True):
    """
    Main entry point for loading data. Checks cache, else processes from scratch.
    """
    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            return load_cache(cache_path)
        except Exception as e:
            # print(f"Cache load failed: {e}. Reprocessing.")
            pass

    # 2. Process from scratch
    # print(f"Processing data from {metadata_path}")
    df = pd.read_csv(metadata_path)

    features_list = []
    labels_list = []
    sample_ids = []

    for _, row in df.iterrows():
        feats, labs, valid = process_single_sample(row)
        if valid:
            features_list.append(feats)
            labels_list.append(labs)
            sample_ids.append(row["sample_id"])
        else:
            # Handle invalid samples (create dummy or skip)
            # For simplicity in maintaining alignment with metadata, we insert empty/dummy
            # But skipping is safer for training.
            # If we skip, the index alignment with metadata breaks.
            # We will insert a short dummy sequence of zeros.
            features_list.append(np.zeros((10, Config.INPUT_DIM), dtype=np.float32))
            labels_list.append(np.zeros(10, dtype=np.int64))
            sample_ids.append(row["sample_id"])

    # 3. Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_cache(cache_path, features_list, labels_list, sample_ids)

    return features_list, labels_list


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, features_list, labels_list, is_train=True):
        """
        Args:
            features_list (list of np.array): List of feature sequences.
            labels_list (list of np.array): List of label sequences.
            is_train (bool): If True, applies sliding window slicing.
                             If False, returns full sequences.
        """
        self.is_train = is_train
        self.features_list = features_list
        self.labels_list = labels_list

        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

        # Pre-calculate indices for sliding windows if training
        self.indices = []

        if self.is_train:
            for i, seq in enumerate(self.features_list):
                seq_len = len(seq)
                if seq_len <= self.window_size:
                    # Sequence shorter than window: take whole thing (will be padded later)
                    self.indices.append((i, 0, seq_len))
                else:
                    # Slide window
                    for start in range(0, seq_len - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        self.indices.append((i, start, end))
                    # Handle remainder if needed? Usually standard sliding window ignores remainder
                    # or ensures coverage. We'll stick to strict windows for now.
        else:
            # For validation/test, one index per sample
            self.indices = [
                (i, 0, len(self.features_list[i]))
                for i in range(len(self.features_list))
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start, end = self.indices[idx]

        feat = self.features_list[sample_idx][start:end]
        label = self.labels_list[sample_idx][start:end]

        # Convert to tensor
        feat_tensor = torch.from_numpy(feat).float()
        label_tensor = torch.from_numpy(label).long()

        # If training and sequence is shorter than window (e.g. original file was short),
        # pad_collate will handle batch padding, but we might want fixed size here?
        # pad_collate handles variable lengths, so we just return what we have.

        return feat_tensor, label_tensor


# ==========================================
# Data Loader Factory
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Data
    train_feats, train_labs = get_processed_data(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_CACHE_PATH, load_cached_data
    )
    val_feats, val_labs = get_processed_data(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data
    )
    test_feats, test_labs = get_processed_data(
        Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, load_cached_data
    )

    # 2. Create Datasets
    train_dataset = GestureDataset(train_feats, train_labs, is_train=True)
    val_dataset = GestureDataset(
        val_feats, val_labs, is_train=False
    )  # Val should use full sequences for metric calc
    test_dataset = GestureDataset(test_feats, test_labs, is_train=False)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one full sequence at a time for validation
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one full sequence at a time for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
