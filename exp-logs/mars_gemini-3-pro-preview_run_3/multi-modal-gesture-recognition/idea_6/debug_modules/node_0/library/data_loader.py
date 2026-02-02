import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# Set fixed seeds
utils.set_seed(config.SEED)


def load_mat_file(path):
    """Safely load .mat file."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def compute_skeleton_features(mat_path):
    """
    Extracts Position, Velocity, and Acceleration from skeleton data.
    Returns: numpy array of shape (NumFrames, 180)
    """
    mat = load_mat_file(mat_path)
    if mat is None or not hasattr(mat, "Video") or not hasattr(mat.Video, "Frames"):
        # Return zeros if data is missing
        # We assume a default length or handle it upstream.
        # Here we return None to indicate failure.
        return None

    frames = mat.Video.Frames
    num_frames = len(frames)

    # 20 joints, 3 coords (X, Y, Z)
    # Shape: (T, 20, 3)
    # Initialize with NaNs to detect missing data later if needed, or zeros
    pos_data = np.zeros(
        (num_frames, config.SKELETON_JOINTS, config.SKELETON_COORDS), dtype=np.float32
    )

    for i, frame in enumerate(frames):
        if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
            wp = frame.Skeleton.WorldPosition
            # Check if WorldPosition is valid (sometimes it's empty or struct)
            try:
                # Assuming wp has X, Y, Z attributes or is an array
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # Some mat files might have multiple skeletons, we take the first tracked one usually
                    # But here WorldPosition seems to be direct coordinates
                    pos_data[i, :, 0] = wp.X
                    pos_data[i, :, 1] = wp.Y
                    pos_data[i, :, 2] = wp.Z
                elif (
                    isinstance(wp, (np.ndarray, list))
                    and len(wp) == config.SKELETON_JOINTS
                ):
                    # Fallback if structure is different but array-like
                    pos_data[i] = wp
            except:
                # If extraction fails for a frame, copy previous frame
                if i > 0:
                    pos_data[i] = pos_data[i - 1]
        else:
            if i > 0:
                pos_data[i] = pos_data[i - 1]

    # Convert mm to meters for stability
    pos_data = pos_data / 1000.0

    # Compute Velocity (First Derivative)
    # V_t = P_t - P_{t-1}
    # Pad first frame with 0
    velocity = np.zeros_like(pos_data)
    velocity[1:] = pos_data[1:] - pos_data[:-1]

    # Compute Acceleration (Second Derivative)
    # A_t = V_t - V_{t-1}
    # Pad first frame with 0 (effectively first 2 frames are affected)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # Flatten features: (T, 20, 9) -> (T, 180)
    # Concatenate along last dimension: Pos, Vel, Acc
    combined = np.concatenate([pos_data, velocity, acceleration], axis=2)
    flattened = combined.reshape(num_frames, -1)

    return flattened


def compute_audio_features(wav_path, target_num_frames):
    """
    Extracts MFCC features and aligns them to video frames.
    Returns: numpy array of shape (NumFrames, N_MFCC)
    """
    if not os.path.exists(wav_path):
        return np.zeros((target_num_frames, config.INPUT_DIM_AUDIO), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(wav_path)

        # Resample if necessary (though config says 16000 is expected)
        if sample_rate != config.AUDIO_SR:
            resampler = T.Resample(sample_rate, config.AUDIO_SR)
            waveform = resampler(waveform)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=config.AUDIO_SR,
            n_mfcc=config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)
        # Shape: (Channels, n_mfcc, time) -> usually (1, 13, T_audio)

        if mfcc.dim() == 3:
            mfcc = mfcc.mean(dim=0)  # Average over channels if stereo

        # mfcc shape: (n_mfcc, time)

        # Interpolate to match video frame count
        # Input to interpolate needs to be (Batch, Channels, Time)
        # We treat n_mfcc as channels
        mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)

        mfcc_aligned = F.interpolate(
            mfcc, size=target_num_frames, mode="linear", align_corners=False
        )

        # Shape: (1, n_mfcc, target_frames) -> (target_frames, n_mfcc)
        mfcc_aligned = mfcc_aligned.squeeze(0).transpose(0, 1)

        return mfcc_aligned.numpy()

    except Exception as e:
        print(f"Error processing audio {wav_path}: {e}")
        return np.zeros((target_num_frames, config.INPUT_DIM_AUDIO), dtype=np.float32)


def process_sample(row):
    """
    Process a single sample row from metadata.
    """
    sample_id = row["sample_id"]
    data_path = os.path.join(config.INPUT_DIR, row["data_path"])
    audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

    # 1. Skeleton Features
    skeleton_feats = compute_skeleton_features(data_path)
    if skeleton_feats is None:
        # Fallback for completely broken files (should be rare/non-existent based on metadata check)
        # Assume a small length to avoid crashes, will likely be filtered or fail gracefully
        print(f"Warning: Failed to load skeleton for {sample_id}")
        return None, None

    num_frames = skeleton_feats.shape[0]

    # 2. Audio Features
    audio_feats = compute_audio_features(audio_path, num_frames)

    # 3. Concatenate (Early Fusion)
    # Skeleton: (T, 180), Audio: (T, 13) -> (T, 193)
    features = np.concatenate([skeleton_feats, audio_feats], axis=1).astype(np.float32)

    # 4. Labels
    # Initialize background (class 0)
    labels = np.zeros(num_frames, dtype=np.int64)

    if "parsed_labels" in row and isinstance(row["parsed_labels"], list):
        for label_info in row["parsed_labels"]:
            # Metadata is 1-based (Matlab convention), convert to 0-based
            # begin is inclusive, end is inclusive in Matlab
            # Python slice: [start, end)
            start_frame = max(0, int(label_info["begin"]) - 1)
            end_frame = min(num_frames, int(label_info["end"]))
            label_id = int(label_info["id"])

            if start_frame < end_frame:
                labels[start_frame:end_frame] = label_id

    return features, labels


def load_data_dict(metadata_path, cache_name, load_cached_data=True):
    """
    Loads data into a dictionary {sample_id: (features, labels)}.
    Uses caching to speed up subsequent runs.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                # Reconstruct dictionary
                # npz stores arrays with keys. We stored 'ids', 'features', 'labels' arrays of objects
                ids = data["ids"]
                features = data["features"]
                labels = data["labels"]

                data_dict = {}
                for i, sample_id in enumerate(ids):
                    data_dict[sample_id] = (features[i], labels[i])
                return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Parse JSON labels if string
    if "labels" in df.columns:
        df["parsed_labels"] = df["labels"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    data_dict = {}
    ids_list = []
    features_list = []
    labels_list = []

    for idx, row in df.iterrows():
        result = process_sample(row)
        if result[0] is not None:
            feats, labs = result
            data_dict[row["sample_id"]] = (feats, labs)

            ids_list.append(row["sample_id"])
            features_list.append(feats)
            labels_list.append(labs)

    # Save to cache
    # We use object array for features/labels because they have variable lengths
    np.savez_compressed(
        cache_path,
        ids=np.array(ids_list),
        features=np.array(features_list, dtype=object),
        labels=np.array(labels_list, dtype=object),
    )
    print(f"Saved cache to {cache_path}")

    return data_dict


class GestureDataset(Dataset):
    def __init__(
        self,
        data_dict,
        mode="train",
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
    ):
        """
        Args:
            data_dict: Dictionary {sample_id: (features, labels)}
            mode: 'train' (sliding windows) or 'val'/'test' (full sequences)
        """
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.data = []

        # Pre-process indices
        sorted_ids = sorted(data_dict.keys())

        if self.mode == "train":
            # Create sliding windows
            for sample_id in sorted_ids:
                feats, labs = data_dict[sample_id]
                num_frames = feats.shape[0]

                # If sequence is shorter than window, pad it?
                # Or just take one window padded.
                if num_frames < self.window_size:
                    # We will handle padding in __getitem__
                    self.data.append(
                        {
                            "sample_id": sample_id,
                            "features": feats,
                            "labels": labs,
                            "start_idx": 0,
                            "is_short": True,
                        }
                    )
                else:
                    # Generate windows
                    for start_idx in range(
                        0, num_frames - self.window_size + 1, self.stride
                    ):
                        self.data.append(
                            {
                                "sample_id": sample_id,
                                "features": feats,  # Store reference, not copy
                                "labels": labs,
                                "start_idx": start_idx,
                                "is_short": False,
                            }
                        )
        else:
            # Full sequences
            for sample_id in sorted_ids:
                feats, labs = data_dict[sample_id]
                self.data.append(
                    {"sample_id": sample_id, "features": feats, "labels": labs}
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == "train":
            feats = item["features"]
            labs = item["labels"]

            if item["is_short"]:
                # Pad short sequence
                seq_len = feats.shape[0]
                pad_len = self.window_size - seq_len

                # Pad features with zeros
                feats_padded = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                # Pad labels with background (0)
                labs_padded = np.pad(
                    labs, (0, pad_len), mode="constant", constant_values=0
                )

                return torch.from_numpy(feats_padded), torch.from_numpy(labs_padded)
            else:
                start = item["start_idx"]
                end = start + self.window_size

                window_feats = feats[start:end]
                window_labs = labs[start:end]

                return torch.from_numpy(window_feats), torch.from_numpy(window_labs)
        else:
            # Return full sequence
            # Note: Batch size must be 1 or use custom collate
            return (
                torch.from_numpy(item["features"]),
                torch.from_numpy(item["labels"]),
                item["sample_id"],
            )


def collate_fn_padd(batch):
    """
    Collate function to handle variable length sequences in validation/test.
    Actually, for TCN/GRU we can just return the list or pad to max in batch.
    However, for simplicity in evaluation loop, we often use batch_size=1.
    If batch_size > 1, we need padding.
    """
    # Check if this is a train batch (tensors) or val batch (tuples with sample_id)
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        # Val/Test mode
        # Sort by length for efficiency (optional)
        # batch.sort(key=lambda x: x[0].shape[0], reverse=True)

        features = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        ids = [x[2] for x in batch]

        # Pad features and labels
        features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0
        )

        # Create mask
        lengths = torch.tensor([x.shape[0] for x in features])

        return features_padded, labels_padded, ids, lengths
    else:
        # Train mode (fixed window size)
        # Default collate is fine
        return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Load raw data dictionaries
    train_dict = load_data_dict(
        config.TRAIN_METADATA_PATH, "train_features", load_cached_data
    )
    val_dict = load_data_dict(
        config.VAL_METADATA_PATH, "val_features", load_cached_data
    )
    test_dict = load_data_dict(
        config.TEST_METADATA_PATH, "test_features", load_cached_data
    )

    # Create Datasets
    train_ds = GestureDataset(train_dict, mode="train")
    val_ds = GestureDataset(val_dict, mode="val")
    test_ds = GestureDataset(
        test_dict, mode="test"
    )  # 'test' mode behaves like 'val' (full seq)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val/Test loaders use batch_size=1 to handle variable lengths easily without complex masking in model
    # Or we can use the collate_fn_padd. Let's use batch_size=1 for safety and simplicity in inference loop.
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader
