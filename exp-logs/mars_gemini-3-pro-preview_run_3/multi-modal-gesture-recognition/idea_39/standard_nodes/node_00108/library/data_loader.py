import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    JOINTS_LIST,
    NUM_JOINTS,
    WINDOW_SIZE,
    STRIDE,
    NUM_MFCC,
    LABEL_MAP,
    BACKGROUND_CLASS_ID,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SEED,
)

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


class PolymorphicParser:
    """
    Handles robust parsing of .mat files with variable structures for Skeleton data.
    """

    @staticmethod
    def parse_skeleton(mat_path):
        try:
            # Load mat file
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            # Access top-level variables as dictionary keys. Cite {debug_lesson_2}
            if "Video" not in mat:
                return None

            video = mat["Video"]

            # Unwrap 0-d NumPy arrays when using squeeze_me=True. Cite {debug_lesson_16}
            if isinstance(video, np.ndarray) and video.ndim == 0:
                video = video.item()

            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames

            # Handle case where Frames is a single object (1 frame video?)
            if not isinstance(frames, (np.ndarray, list)):
                frames = [frames]

            num_frames = len(frames)
            # Shape: (T, 20, 3)
            skeleton_data = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)

            for t, frame in enumerate(frames):
                if not hasattr(frame, "Skeleton"):
                    continue

                skel = frame.Skeleton

                # Case A: Skeleton is an array of 20 joint objects
                if isinstance(skel, (np.ndarray, list)) and len(skel) == NUM_JOINTS:
                    for j, joint in enumerate(skel):
                        skeleton_data[t, j] = PolymorphicParser._extract_xyz(joint)

                # Case B: Skeleton is an object with joint names as fields
                elif isinstance(skel, object):
                    # Check if it has fields matching JOINTS_LIST
                    # Some datasets might have slightly different casing, but we try exact match first
                    valid_joint_found = False
                    for j, joint_name in enumerate(JOINTS_LIST):
                        if hasattr(skel, joint_name):
                            joint_obj = getattr(skel, joint_name)
                            skeleton_data[t, j] = PolymorphicParser._extract_xyz(
                                joint_obj
                            )
                            valid_joint_found = True

                    # If not found by name, maybe it's a single object wrapping an array?
                    # Fallback logic could go here, but usually it's one of the above.
                    if not valid_joint_found:
                        # Attempt to treat as array if it was squeezed weirdly
                        pass

            return skeleton_data

        except Exception as e:
            print(f"Error parsing {mat_path}: {e}")
            return None

    @staticmethod
    def _extract_xyz(joint_obj):
        # Helper to extract X, Y, Z from a joint object or structure
        if hasattr(joint_obj, "WorldPosition"):
            wp = joint_obj.WorldPosition
            if isinstance(wp, (np.ndarray, list)) and len(wp) >= 3:
                return np.array(wp[:3], dtype=np.float32)
            elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                return np.array([wp.X, wp.Y, wp.Z], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)


class AudioProcessor:
    """
    Handles loading and MFCC extraction for audio files.
    """

    @staticmethod
    def process_audio(audio_path, target_num_frames):
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            # We use default settings but ensure n_mfcc matches config
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=NUM_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # Shape: (Channel, n_mfcc, time)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = torch.mean(mfcc, dim=0, keepdim=True)

            # mfcc shape: (1, n_mfcc, time_steps)
            mfcc = mfcc.squeeze(0).numpy().T  # (time_steps, n_mfcc)

            # Interpolate to match video frames
            current_len = mfcc.shape[0]
            if current_len != target_num_frames:
                # Use simple linear interpolation
                x_old = np.linspace(0, 1, current_len)
                x_new = np.linspace(0, 1, target_num_frames)

                mfcc_resampled = np.zeros(
                    (target_num_frames, NUM_MFCC), dtype=np.float32
                )
                for c in range(NUM_MFCC):
                    mfcc_resampled[:, c] = np.interp(x_new, x_old, mfcc[:, c])
                return mfcc_resampled

            return mfcc.astype(np.float32)

        except Exception as e:
            # Return zeros if audio fails
            return np.zeros((target_num_frames, NUM_MFCC), dtype=np.float32)


class KinematicAugmentor:
    """
    Applies kinematic augmentation and derives velocity/acceleration.
    """

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, position_data):
        """
        Args:
            position_data: (T, J, 3) numpy array
        Returns:
            features: (T, J * 9) numpy array [Pos, Vel, Acc] flattened
        """
        T, J, C = position_data.shape
        data = position_data.copy()

        if self.augment:
            # Random Rotation around Y-axis
            theta = np.deg2rad(np.random.uniform(-20, 20))
            c, s = np.cos(theta), np.sin(theta)

            # Apply rotation: x' = x*c - z*s, z' = x*s + z*c
            x = data[:, :, 0]
            z = data[:, :, 2]
            data[:, :, 0] = x * c - z * s
            data[:, :, 2] = x * s + z * c

            # Random Scale
            scale = np.random.uniform(0.9, 1.1)
            data = data * scale

        # Derive Velocity (pad first frame)
        # V[t] = P[t] - P[t-1]
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]

        # Derive Acceleration (pad first frame)
        # A[t] = V[t] - V[t-1]
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]

        # Concatenate: (T, J, 9)
        combined = np.concatenate([data, velocity, acceleration], axis=2)

        # Flatten joints: (T, J*9)
        return combined.reshape(T, -1)


class GestureDataset(Dataset):
    def __init__(
        self, samples_dict, augment=False, window_size=WINDOW_SIZE, stride=STRIDE
    ):
        """
        Args:
            samples_dict: Dictionary containing processed data for each sample.
                          Format: {sample_id: {'features': (T, D), 'labels': (T,)}}
            augment: Boolean, whether to apply kinematic augmentation.
            window_size: Int, sliding window size.
            stride: Int, stride for sliding window.
        """
        self.samples = samples_dict
        self.augment = augment
        self.window_size = window_size
        self.stride = stride
        self.augmentor = KinematicAugmentor(augment=augment)

        # Pre-calculate window indices
        self.windows = []
        self.sample_ids = sorted(list(self.samples.keys()))

        for sid in self.sample_ids:
            num_frames = self.samples[sid]["skeleton"].shape[0]
            # Generate windows
            # If sequence is shorter than window, pad or skip?
            # Given dataset stats (avg ~20 gestures, long videos), usually T >> window_size.
            # However, we handle edge cases.
            if num_frames < window_size:
                # Pad logic handled in getitem if needed, or just skip small files (unlikely here)
                self.windows.append((sid, 0))
            else:
                for start in range(0, num_frames - window_size + 1, stride):
                    self.windows.append((sid, start))

                # Ensure last frame is covered if not exact fit
                if (num_frames - window_size) % stride != 0:
                    self.windows.append((sid, num_frames - window_size))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sid, start_frame = self.windows[idx]
        sample = self.samples[sid]

        # Raw Data
        skel_raw = sample["skeleton"]  # (T, J, 3)
        audio_raw = sample["audio"]  # (T, MFCC)
        labels_raw = sample["labels"]  # (T,)

        end_frame = start_frame + self.window_size

        # Handle padding for short sequences
        curr_len = skel_raw.shape[0]
        if curr_len < self.window_size:
            # Pad with zeros
            pad_len = self.window_size - curr_len
            skel_window = np.pad(
                skel_raw, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
            )
            audio_window = np.pad(audio_raw, ((0, pad_len), (0, 0)), mode="constant")
            labels_window = np.pad(
                labels_raw,
                (0, pad_len),
                mode="constant",
                constant_values=BACKGROUND_CLASS_ID,
            )
        else:
            skel_window = skel_raw[start_frame:end_frame]
            audio_window = audio_raw[start_frame:end_frame]
            labels_window = labels_raw[start_frame:end_frame]

        # Apply Kinematic Augmentation / Derivation
        # Returns (Window, J*9)
        kinematic_features = self.augmentor(skel_window)

        # Concatenate Audio
        # Final shape: (Window, J*9 + MFCC)
        features = np.concatenate([kinematic_features, audio_window], axis=1)

        return torch.FloatTensor(features), torch.LongTensor(labels_window)


def load_and_process_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads raw data, parses, aligns, and caches it.
    Returns a dictionary of processed samples.
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {cache_name} from cache...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary
            data_dict = {}
            # Keys in npz are flattened, e.g., 'Sample001_skeleton'
            # We need to group them back
            keys = sorted(loaded.files)
            sample_ids = set(k.split("_")[0] for k in keys)

            for sid in sample_ids:
                data_dict[sid] = {
                    "skeleton": loaded[f"{sid}_skeleton"],
                    "audio": loaded[f"{sid}_audio"],
                    "labels": loaded[f"{sid}_labels"],
                }
            return data_dict
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {cache_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    # Parse JSON labels
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    processed_data = {}

    for idx, row in df.iterrows():
        sid = row["sample_id"]
        mat_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # Parse Skeleton
        skeleton = PolymorphicParser.parse_skeleton(mat_path)
        if skeleton is None:
            continue  # Skip corrupt samples

        num_frames = skeleton.shape[0]

        # Parse Audio
        audio = AudioProcessor.process_audio(audio_path, num_frames)

        # Build Labels
        labels = np.full(num_frames, BACKGROUND_CLASS_ID, dtype=np.int64)
        for ann in row["parsed_labels"]:
            gid = ann["id"]
            start = max(0, ann["begin"] - 1)  # 1-based to 0-based
            end = min(num_frames, ann["end"])
            labels[start:end] = gid

        processed_data[sid] = {"skeleton": skeleton, "audio": audio, "labels": labels}

    # 3. Save to cache
    save_dict = {}
    for sid, data in processed_data.items():
        save_dict[f"{sid}_skeleton"] = data["skeleton"]
        save_dict[f"{sid}_audio"] = data["audio"]
        save_dict[f"{sid}_labels"] = data["labels"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {cache_name} to {cache_path}")

    return processed_data


def get_dataloaders(batch_size=32, load_cached=True):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    # Load Data
    train_data = load_and_process_data(
        TRAIN_METADATA_PATH, "dataset_train", load_cached
    )
    val_data = load_and_process_data(VAL_METADATA_PATH, "dataset_val", load_cached)

    # Create Datasets
    train_dataset = GestureDataset(train_data, augment=True)
    val_dataset = GestureDataset(val_data, augment=False)

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, load_cached=True):
    """
    Entry point for Test DataLoader.
    """
    test_data = load_and_process_data(TEST_METADATA_PATH, "dataset_test", load_cached)
    test_dataset = GestureDataset(test_data, augment=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return (
        test_loader,
        test_data,
    )  # Return raw data dict too for reconstruction if needed
