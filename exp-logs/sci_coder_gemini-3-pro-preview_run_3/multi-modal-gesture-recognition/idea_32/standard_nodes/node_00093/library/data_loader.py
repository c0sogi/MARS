import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library import config

# Ensure reproducible results
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


class PolymorphicParser:
    """
    Parses .mat files robustly, handling different structural variations
    produced by MATLAB export and scipy.io loading.
    """

    JOINTS_ORDER = [
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

    @staticmethod
    def parse_skeleton(mat_path):
        """
        Extracts skeleton data (T, 20, 3) from a .mat file.
        Returns numpy array of shape (NumFrames, 20, 3) in millimeters.
        """
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None

        if "Video" not in mat:
            return None

        video = mat["Video"]
        # Unwrap 0-d array if present (due to squeeze_me=True)
        if isinstance(video, np.ndarray) and video.ndim == 0:
            video = video.item()

        # Determine number of frames
        num_frames = 0
        if hasattr(video, "NumFrames"):
            num_frames = int(video.NumFrames)
        elif hasattr(video, "Frames"):
            num_frames = len(video.Frames)

        if num_frames == 0:
            return None

        # Initialize skeleton array (T, 20, 3)
        # Fill with NaNs to detect missing data later
        skeleton_data = np.full((num_frames, 20, 3), np.nan, dtype=np.float32)

        if not hasattr(video, "Frames"):
            return np.nan_to_num(skeleton_data, nan=0.0)

        frames = video.Frames

        # Handle case where Frames is a single object (1 frame)
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        # Iterate over frames
        for t, frame in enumerate(frames):
            if t >= num_frames:
                break

            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Helper to extract pos from a joint object
            def get_pos(joint_obj):
                if hasattr(joint_obj, "WorldPosition"):
                    wp = joint_obj.WorldPosition
                    # Check if wp is an object with X,Y,Z or array
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        return [float(wp.X), float(wp.Y), float(wp.Z)]
                    elif isinstance(wp, (np.ndarray, list)) and len(wp) >= 3:
                        return [float(wp[0]), float(wp[1]), float(wp[2])]
                return [np.nan, np.nan, np.nan]

            # Case 1: Skeleton is an array of joints (common case)
            if isinstance(skel, (list, np.ndarray)):
                # Try to match by JointsType if available, else assume order
                if len(skel) == 20:
                    # Check if first element has JointsType
                    if hasattr(skel[0], "JointsType"):
                        # Map by name
                        for joint in skel:
                            j_type = str(joint.JointsType)
                            if j_type in PolymorphicParser.JOINTS_ORDER:
                                idx = PolymorphicParser.JOINTS_ORDER.index(j_type)
                                skeleton_data[t, idx] = get_pos(joint)
                    else:
                        # Assume order
                        for j_idx in range(20):
                            skeleton_data[t, j_idx] = get_pos(skel[j_idx])

            # Case 2: Skeleton is a single object (less common in this dataset)
            elif isinstance(skel, scipy.io.matlab.mat_struct):
                pass

        # Interpolate missing frames (NaNs)
        flat_skel = skeleton_data.reshape(num_frames, -1)
        df_skel = pd.DataFrame(flat_skel)
        # Linear interpolate
        df_skel = df_skel.interpolate(method="linear", limit_direction="both", axis=0)
        # Fill remaining NaNs with 0
        df_skel = df_skel.fillna(0.0)

        skeleton_data = df_skel.values.reshape(num_frames, 20, 3)

        return skeleton_data.astype(np.float32)


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
            data_dict: Dictionary containing 'skeletons', 'audio', 'labels', 'ids'.
            mode: 'train', 'val', or 'test'.
            window_size: Length of sliding window.
            stride: Stride for sliding window.
        """
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        self.skeletons = data_dict["skeletons"]  # List of (T, 20, 3)
        self.audio = data_dict["audio"]  # List of (T, n_mfcc)
        self.labels = data_dict["labels"]  # List of (T,)
        self.ids = data_dict["ids"]  # List of strings

        # Pre-calculate windows
        self.windows = []
        for sample_idx, (skel, aud, lbl) in enumerate(
            zip(self.skeletons, self.audio, self.labels)
        ):
            num_frames = skel.shape[0]

            # Ensure audio and skeleton are aligned in length
            min_len = min(skel.shape[0], aud.shape[0], lbl.shape[0])

            if num_frames < self.window_size:
                # Pad logic handled in __getitem__
                self.windows.append((sample_idx, 0))
            else:
                # Sliding window
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.windows.append((sample_idx, start))

                # Ensure we cover the end of the sequence for test/val
                if mode != "train":
                    last_start = num_frames - self.window_size
                    if last_start > 0 and (last_start % self.stride != 0):
                        self.windows.append((sample_idx, last_start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.windows[idx]

        # Get full sequences
        full_skel = self.skeletons[sample_idx]  # (T, 20, 3)
        full_audio = self.audio[sample_idx]  # (T, n_mfcc)
        full_label = self.labels[sample_idx]  # (T,)

        seq_len = full_skel.shape[0]
        end_frame = start_frame + self.window_size

        # Extract slices
        if seq_len < self.window_size:
            # Padding required
            pad_len = self.window_size - seq_len

            skel_slice = full_skel
            audio_slice = full_audio
            label_slice = full_label

            # Pad with zeros (or edge values)
            skel_pad = np.zeros((pad_len, 20, 3), dtype=np.float32)
            skel_window = np.concatenate([skel_slice, skel_pad], axis=0)

            audio_pad = np.zeros((pad_len, full_audio.shape[1]), dtype=np.float32)
            audio_window = np.concatenate([audio_slice, audio_pad], axis=0)

            label_pad = np.zeros((pad_len,), dtype=np.int64)
            label_window = np.concatenate([label_slice, label_pad], axis=0)

        else:
            skel_window = full_skel[start_frame:end_frame]
            audio_window = full_audio[start_frame:end_frame]
            label_window = full_label[start_frame:end_frame]

        # --- Augmentation (Train Only) ---
        if self.mode == "train":
            # 1. Random Scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            skel_window = skel_window * scale

            # 2. Random Rotation around Y-axis (-20 to +20 degrees)
            theta = np.deg2rad(np.random.uniform(-20, 20))
            c, s = np.cos(theta), np.sin(theta)

            x = skel_window[:, :, 0]
            y = skel_window[:, :, 1]
            z = skel_window[:, :, 2]

            x_new = x * c + z * s
            z_new = -x * s + z * c

            skel_window = np.stack([x_new, y, z_new], axis=2)

        # --- Kinematics (V, A) ---
        # Compute Velocity (diff)
        vel = np.zeros_like(skel_window)
        vel[1:] = skel_window[1:] - skel_window[:-1]

        # Compute Acceleration (diff of vel)
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate Skeleton Features: (W, 20, 9)
        # Flatten joints: (W, 180)
        skel_feats = np.concatenate([skel_window, vel, acc], axis=2)
        skel_feats = skel_feats.reshape(self.window_size, -1)

        # --- Fusion ---
        # Concatenate Audio: (W, 180 + n_mfcc)
        features = np.concatenate([skel_feats, audio_window], axis=1)

        # Convert to Tensor
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(label_window).long()

        return features, labels


def process_audio(audio_path, target_frames):
    """
    Loads audio, computes MFCC, and aligns to target_frames.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except:
        # Return silent features if audio missing
        return np.zeros((target_frames, 13), dtype=np.float32)

    # Resample if necessary (target 16kHz)
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Compute MFCC
    mfcc_transform = T.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
    mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

    # Align to video frames using interpolation
    mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)

    mfcc_aligned = torch.nn.functional.interpolate(
        mfcc, size=target_frames, mode="linear", align_corners=False
    )

    # (1, n_mfcc, W) -> (W, n_mfcc)
    mfcc_aligned = mfcc_aligned.squeeze(0).permute(1, 0).numpy()

    return mfcc_aligned.astype(np.float32)


def load_and_cache_data(metadata_path, cache_path, load_cached_data=True):
    """
    Loads data from metadata CSV. Uses caching.
    """
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            return {
                "skeletons": loaded["skeletons"],
                "audio": loaded["audio"],
                "labels": loaded["labels"],
                "ids": loaded["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Parse labels JSON
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    skeletons_list = []
    audio_list = []
    labels_list = []
    ids_list = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(config.INPUT_DIR, row["audio_path"])

        # 1. Load Skeleton
        skeleton = PolymorphicParser.parse_skeleton(data_path)
        if skeleton is None:
            continue

        num_frames = skeleton.shape[0]

        # 2. Load Audio and Align
        audio_feats = process_audio(audio_path, num_frames)

        # 3. Create Frame-wise Labels
        label_seq = np.zeros(num_frames, dtype=np.int64)

        for gesture in row["parsed_labels"]:
            gid = gesture["id"]
            # MATLAB 1-based indexing to 0-based
            start = max(0, gesture["begin"] - 1)
            end = min(num_frames, gesture["end"])
            label_seq[start:end] = gid

        skeletons_list.append(skeleton)
        audio_list.append(audio_feats)
        labels_list.append(label_seq)
        ids_list.append(sample_id)

    # Save to cache
    np.savez_compressed(
        cache_path,
        skeletons=np.array(skeletons_list, dtype=object),
        audio=np.array(audio_list, dtype=object),
        labels=np.array(labels_list, dtype=object),
        ids=np.array(ids_list, dtype=object),
    )

    return {
        "skeletons": skeletons_list,
        "audio": audio_list,
        "labels": labels_list,
        "ids": ids_list,
    }


def get_data_loaders(load_cached_data=True):
    """
    Factory function to create dataloaders.
    """
    train_cache = os.path.join(config.CACHE_DIR, "dataset_train.npz")
    val_cache = os.path.join(config.CACHE_DIR, "dataset_val.npz")
    test_cache = os.path.join(config.CACHE_DIR, "dataset_test.npz")

    train_data = load_and_cache_data(
        config.TRAIN_METADATA_PATH, train_cache, load_cached_data
    )
    val_data = load_and_cache_data(
        config.VAL_METADATA_PATH, val_cache, load_cached_data
    )
    test_data = load_and_cache_data(
        config.TEST_METADATA_PATH, test_cache, load_cached_data
    )

    train_dataset = GestureDataset(train_data, mode="train")
    val_dataset = GestureDataset(val_data, mode="val")
    test_dataset = GestureDataset(test_data, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_test_sequences(load_cached_data=True):
    """
    Returns the raw test data (not windowed) for final inference aggregation.
    """
    test_cache = os.path.join(config.CACHE_DIR, "dataset_test.npz")
    test_data = load_and_cache_data(
        config.TEST_METADATA_PATH, test_cache, load_cached_data
    )
    return test_data
