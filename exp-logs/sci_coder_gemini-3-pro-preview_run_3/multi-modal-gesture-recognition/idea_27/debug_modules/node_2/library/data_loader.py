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

# ==========================================
# 1. Robust Parsing & Feature Extraction
# ==========================================


class RobustMatParser:
    """
    Parses .mat files robustly, handling inconsistent struct/cell array formats.
    """

    @staticmethod
    def load(mat_path):
        try:
            # Load with squeeze_me=True to simplify structure access
            mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            return mat
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None

    @staticmethod
    def parse_skeleton(mat, num_frames):
        """
        Extracts skeleton joints (T, 20, 3) from the loaded mat object.
        """
        # Default container: T frames, 20 joints, 3 coords
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        if not hasattr(mat, "Video") or not hasattr(mat.Video, "Frames"):
            return skeleton_data

        frames = mat.Video.Frames

        # Handle case where Frames is a single object (1 frame) vs array
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        # Iterate through frames
        for t, frame in enumerate(frames):
            if t >= num_frames:
                break

            # Check if Skeleton exists and is valid
            if hasattr(frame, "Skeleton") and frame.Skeleton is not None:
                skel = frame.Skeleton

                # Handle Skeleton being an array (multiple users) -> take first valid
                if isinstance(skel, (list, np.ndarray)):
                    if len(skel) > 0:
                        skel = skel[0]
                    else:
                        continue  # Empty skeleton array

                # Extract Joint Positions
                # Expecting WorldPosition with X, Y, Z
                if hasattr(skel, "WorldPosition"):
                    # WorldPosition might be an array of structs (one per joint)
                    # The prompt implies structure: Skeleton.WorldPosition.X (if single)
                    # or Skeleton is array of joints?
                    # Prompt says: "Skeleton Frame: An array of Skeleton structures... contains joint positions"
                    # Actually usually in these datasets: frame.Skeleton is a struct containing JointsType and WorldPosition arrays.
                    # Let's try to adapt to common Kinect formats.

                    # Strategy: Check if WorldPosition is an array of length 20
                    wp = skel.WorldPosition

                    if (
                        isinstance(wp, (list, np.ndarray))
                        and len(wp) == Config.NUM_JOINTS
                    ):
                        for j_idx in range(Config.NUM_JOINTS):
                            joint = wp[j_idx]
                            if (
                                hasattr(joint, "X")
                                and hasattr(joint, "Y")
                                and hasattr(joint, "Z")
                            ):
                                skeleton_data[t, j_idx, 0] = joint.X
                                skeleton_data[t, j_idx, 1] = joint.Y
                                skeleton_data[t, j_idx, 2] = joint.Z

            # Simple imputation: if frame is all zeros (and not first frame), copy previous
            if t > 0 and np.all(skeleton_data[t] == 0):
                skeleton_data[t] = skeleton_data[t - 1]

        return skeleton_data


class FeatureExtractor:
    """
    Computes Audio MFCCs and Kinematic Features.
    """

    @staticmethod
    def process_audio(audio_path, target_num_frames):
        """
        Loads audio, computes MFCC, and resizes to match video frame count.
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            # We use a window length approx 25ms, hop length 10ms standard,
            # but we will resize anyway.
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = mfcc_transform(waveform)  # (Channel, n_mfcc, time)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = mfcc.mean(dim=0, keepdim=True)

            # mfcc shape: (1, n_mfcc, time_steps)

            # Resize to match video frames (target_num_frames)
            # Input to interpolate must be (Batch, Channels, Time)
            mfcc = F.interpolate(
                mfcc.unsqueeze(0),
                size=target_num_frames,
                mode="linear",
                align_corners=False,
            )

            # Output: (target_num_frames, n_mfcc)
            return mfcc.squeeze(0).squeeze(0).permute(1, 0).numpy()

        except Exception:
            # Return zeros if audio fails
            return np.zeros((target_num_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    @staticmethod
    def compute_kinematics(positions):
        """
        Computes Velocity and Acceleration from positions.
        positions: (T, J, 3)
        Returns: (T, J, 3) velocity, (T, J, 3) acceleration
        """
        # Velocity: P[t] - P[t-1]
        # Pad first frame with 0
        vel = np.diff(positions, axis=0, prepend=positions[0:1])

        # Acceleration: V[t] - V[t-1]
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        return vel, acc


class KinematicAugmentor:
    """
    Applies 3D augmentations to raw skeleton data.
    """

    @staticmethod
    def augment(positions):
        """
        positions: (T, J, 3)
        Returns: augmented positions
        """
        T, J, C = positions.shape
        aug_pos = positions.copy()

        # 1. Random Rotation around Y-axis
        theta = np.random.uniform(-np.pi / 9, np.pi / 9)  # +/- 20 degrees
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # Apply rotation to all points
        # Reshape to (T*J, 3) for matmul
        flat_pos = aug_pos.reshape(-1, 3)
        flat_pos = np.dot(flat_pos, rotation_matrix.T)
        aug_pos = flat_pos.reshape(T, J, 3)

        # 2. Random Scaling (0.9 to 1.1)
        scale = np.random.uniform(0.9, 1.1)
        aug_pos = aug_pos * scale

        return aug_pos


# ==========================================
# 2. Dataset Logic
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, samples_dict, mode="train"):
        """
        samples_dict: Dictionary containing arrays:
            - 'positions': (TotalFrames, 20, 3)
            - 'audio': (TotalFrames, 13)
            - 'labels': (TotalFrames,)
            - 'seq_lens': (NumSamples,)
            - 'ids': (NumSamples,)
        mode: 'train', 'val', or 'test'
        """
        self.mode = mode
        self.positions = samples_dict["positions"]
        self.audio = samples_dict["audio"]
        self.labels = samples_dict["labels"]
        self.seq_lens = samples_dict["seq_lens"]
        self.ids = samples_dict["ids"]

        # Calculate cumulative indices to slice the flattened arrays
        self.seq_starts = np.concatenate(([0], np.cumsum(self.seq_lens)[:-1]))

        # Generate Sliding Windows
        self.windows = []
        stride = Config.STRIDE_TRAIN if mode == "train" else Config.STRIDE_TEST

        for i, length in enumerate(self.seq_lens):
            # For each sequence, generate window start indices
            # If sequence is shorter than window, we take one window (0) and pad later
            if length <= Config.WINDOW_SIZE:
                self.windows.append((i, 0))
            else:
                # Generate starts
                starts = range(0, length - Config.WINDOW_SIZE + 1, stride)
                for s in starts:
                    self.windows.append((i, s))

                # Ensure the last frame is covered (add a final window ending at length)
                if (length - Config.WINDOW_SIZE) % stride != 0:
                    self.windows.append((i, length - Config.WINDOW_SIZE))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_frame = self.windows[idx]

        # Retrieve full sequence data
        seq_start_idx = self.seq_starts[seq_idx]
        seq_len = self.seq_lens[seq_idx]

        # Slicing from the flattened arrays
        # We extract the specific window directly if possible, but for kinematics/aug
        # we might want context. However, for efficiency, we extract the window + context
        # or just the window.
        # To compute kinematics correctly (finite diff), we need 1 frame previous context.
        # But since we operate on windows, we can just compute on the window and pad the first diff.
        # Augmentation is rotation, which is point-wise, so window is fine.

        # Define extraction range
        end_frame = start_frame + Config.WINDOW_SIZE

        # Handle padding for short sequences
        pad_len = 0
        if seq_len < Config.WINDOW_SIZE:
            extract_len = seq_len
            pad_len = Config.WINDOW_SIZE - seq_len
            end_frame = start_frame + extract_len
        else:
            extract_len = Config.WINDOW_SIZE

        # Edge case: Empty sequence (Cite debug_lesson_20)
        if extract_len == 0:
            features = torch.zeros(
                (Config.WINDOW_SIZE, Config.INPUT_DIM), dtype=torch.float32
            )
            labels = torch.full(
                (Config.WINDOW_SIZE,), Config.BACKGROUND_CLASS_ID, dtype=torch.long
            )
            return features, labels

        # Extract Raw Data
        abs_start = seq_start_idx + start_frame
        abs_end = seq_start_idx + end_frame

        pos_window = self.positions[abs_start:abs_end].copy()  # (T, 20, 3)
        audio_window = self.audio[abs_start:abs_end].copy()  # (T, 13)
        label_window = self.labels[abs_start:abs_end].copy()  # (T,)

        # 1. Augmentation (Train only)
        if self.mode == "train":
            pos_window = KinematicAugmentor.augment(pos_window)

        # 2. Compute Kinematics
        # Note: units are mm. Convert to meters for numerical stability.
        pos_window = pos_window / 1000.0
        vel_window, acc_window = FeatureExtractor.compute_kinematics(pos_window)

        # 3. Flatten Skeleton Features
        # (T, 20, 3) -> (T, 60)
        pos_flat = pos_window.reshape(extract_len, -1)
        vel_flat = vel_window.reshape(extract_len, -1)
        acc_flat = acc_window.reshape(extract_len, -1)

        # 4. Concatenate Features
        # [Pos, Vel, Acc, Audio] -> (T, 60+60+60+13) = (T, 193)
        features = np.concatenate([pos_flat, vel_flat, acc_flat, audio_window], axis=1)

        # 5. Padding (if sequence was shorter than window)
        if pad_len > 0:
            # Pad features with 0
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            # Pad labels with Background (0)
            label_window = np.pad(
                label_window,
                (0, pad_len),
                mode="constant",
                constant_values=Config.BACKGROUND_CLASS_ID,
            )

        # Convert to Tensor
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(label_window)

        return features, labels


# ==========================================
# 3. Data Loading Pipeline
# ==========================================


def process_and_cache_data(metadata_path, cache_file, load_cached_data=True):
    """
    Processes raw data listed in metadata_path and caches it to cache_file.
    Uses np.savez to avoid pickle.
    """
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            return np.load(cache_file)
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    list_pos = []
    list_audio = []
    list_labels = []
    list_lens = []
    list_ids = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # 1. Parse Skeleton
        mat = RobustMatParser.load(data_path)
        if mat is None:
            continue

        # Get NumFrames
        if hasattr(mat, "Video") and hasattr(mat.Video, "NumFrames"):
            num_frames = int(mat.Video.NumFrames)
        else:
            # Fallback if NumFrames missing
            num_frames = 0

        skeleton = RobustMatParser.parse_skeleton(mat, num_frames)
        real_frames = skeleton.shape[0]

        # 2. Process Audio
        audio_mfcc = FeatureExtractor.process_audio(audio_path, real_frames)

        # 3. Process Labels
        # Initialize with Background (0)
        label_seq = np.zeros(real_frames, dtype=np.int64)

        # Parse JSON labels
        try:
            labels_meta = (
                json.loads(row["labels"]) if isinstance(row["labels"], str) else []
            )
            for l in labels_meta:
                start = max(0, int(l["begin"]) - 1)  # 1-based to 0-based
                end = min(real_frames, int(l["end"]))
                lid = int(l["id"])
                if lid > 0 and start < end:
                    label_seq[start:end] = lid
        except:
            pass

        list_pos.append(skeleton)
        list_audio.append(audio_mfcc)
        list_labels.append(label_seq)
        list_lens.append(real_frames)
        list_ids.append(sample_id)

    # Flatten for storage
    if len(list_pos) > 0:
        all_pos = np.concatenate(list_pos, axis=0)
        all_audio = np.concatenate(list_audio, axis=0)
        all_labels = np.concatenate(list_labels, axis=0)
        all_lens = np.array(list_lens, dtype=np.int32)
        all_ids = np.array(list_ids, dtype=str)
    else:
        # Empty fallback
        all_pos = np.zeros((0, 20, 3), dtype=np.float32)
        all_audio = np.zeros((0, 13), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)
        all_lens = np.zeros((0,), dtype=np.int32)
        all_ids = np.zeros((0,), dtype=str)

    # Save to cache
    np.savez_compressed(
        cache_file,
        positions=all_pos,
        audio=all_audio,
        labels=all_labels,
        seq_lens=all_lens,
        ids=all_ids,
    )
    print(f"Saved processed data to {cache_file}")

    # Reload to return the NpzFile object
    return np.load(cache_file)


def get_dataloaders(load_cached_data=True):
    """
    Main entry point. Returns train, val, test dataloaders.
    """
    # Paths
    train_meta = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta = os.path.join(Config.METADATA_DIR, "test.csv")

    cache_train = os.path.join(Config.CACHE_DIR, "dataset_train.npz")
    cache_val = os.path.join(Config.CACHE_DIR, "dataset_val.npz")
    cache_test = os.path.join(Config.CACHE_DIR, "dataset_test.npz")

    # Process/Load Data
    data_train = process_and_cache_data(train_meta, cache_train, load_cached_data)
    data_val = process_and_cache_data(val_meta, cache_val, load_cached_data)
    data_test = process_and_cache_data(test_meta, cache_test, load_cached_data)

    # Create Datasets
    ds_train = GestureDataset(data_train, mode="train")
    ds_val = GestureDataset(data_val, mode="val")
    ds_test = GestureDataset(data_test, mode="test")

    # Create DataLoaders
    dl_train = DataLoader(
        ds_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val and Test: Shuffle=False to keep order (though windowing breaks sequence order in batch,
    # reconstruction relies on sample IDs in inference loop, but here we just return standard loaders)
    dl_val = DataLoader(
        ds_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return dl_train, dl_val, dl_test
