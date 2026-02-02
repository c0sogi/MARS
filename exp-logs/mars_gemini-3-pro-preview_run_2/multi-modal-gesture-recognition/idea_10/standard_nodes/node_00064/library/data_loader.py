import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior for libraries
set_seed(Config.SEED)


def load_multimodal_data(row, input_dir):
    """
    Loads and processes a single sample's multimodal data (Skeleton + Audio + Labels).

    Args:
        row (pd.Series): Row from the metadata DataFrame.
        input_dir (str): Base input directory.

    Returns:
        dict: Dictionary containing 'skeleton', 'audio', 'labels', 'num_frames'.
              Returns None if loading fails.
    """
    sample_id = row["sample_id"]
    data_path = os.path.join(input_dir, row["data_path"])
    audio_path = os.path.join(input_dir, row["audio_path"])

    # -------------------------------------------------------------------------
    # 1. Load Skeleton Data & Labels from .mat
    # -------------------------------------------------------------------------
    try:
        mat = scipy.io.loadmat(data_path, struct_as_record=False, squeeze_me=True)
        if "Video" not in mat:
            return None
        video = mat["Video"]

        # Extract Frame Count
        num_frames = int(getattr(video, "NumFrames", 0))
        if num_frames == 0:
            return None

        # Extract Skeleton Joints
        # Video.Frames is an array of structs. We need to iterate or vectorize.
        # Structure: Video.Frames[i].Skeleton.WorldPosition.{X,Y,Z}
        # This can be slow, but necessary given the format.
        frames_struct = getattr(video, "Frames", [])

        # Pre-allocate skeleton array: (T, Joints, 3)
        # Config.SELECTED_JOINTS defines which joints to keep.
        # The raw data has 20 joints.
        num_raw_joints = 20
        skeleton_data = np.zeros((num_frames, num_raw_joints, 3), dtype=np.float32)

        # Handle cases where frames_struct is a single object or array
        if not isinstance(frames_struct, np.ndarray):
            frames_struct = np.array([frames_struct])

        # Check if we have enough frames in the struct
        actual_frames = len(frames_struct)
        if actual_frames != num_frames:
            # Resize to match actual data availability
            num_frames = actual_frames
            skeleton_data = skeleton_data[:num_frames]

        for t in range(num_frames):
            try:
                frame_obj = frames_struct[t]
                # Check if Skeleton exists and track state is valid (implied by existence usually)
                skel_obj = getattr(frame_obj, "Skeleton", None)

                if skel_obj is not None:
                    # skel_obj.WorldPosition is an array of structs or a single struct?
                    # Description says: "Skeleton... contains joint positions... WorldPosition"
                    # Usually in these datasets, WorldPosition is an array of size NumJoints
                    # OR there are multiple skeletons. We assume single user or take the first.

                    # Handling potential multi-user tracking:
                    # The dataset description mentions "User Index".
                    # We will assume the main skeleton is the one recorded in Video.Frames.Skeleton.

                    # If Skeleton is an array (multiple users), take the first one
                    if isinstance(skel_obj, np.ndarray):
                        if skel_obj.size > 0:
                            skel_obj = skel_obj[0]
                        else:
                            continue  # No skeleton

                    # Now we have a single Skeleton structure
                    # It contains WorldPosition which might be an array of joints
                    # OR we have to access joints by name/index.
                    # Description: "JointsType... WorldPosition... X, Y, Z"
                    # Usually WorldPosition is an array of 20 structs.

                    world_pos = getattr(skel_obj, "WorldPosition", None)
                    if (
                        world_pos is not None
                        and isinstance(world_pos, np.ndarray)
                        and len(world_pos) >= num_raw_joints
                    ):
                        for j in range(num_raw_joints):
                            # Each joint has X, Y, Z
                            joint = world_pos[j]
                            skeleton_data[t, j, 0] = float(joint.X)
                            skeleton_data[t, j, 1] = float(joint.Y)
                            skeleton_data[t, j, 2] = float(joint.Z)
            except Exception:
                # If frame parsing fails, leave as zeros (padding/missing)
                pass

        # Filter selected joints
        # skeleton_data: (T, 20, 3) -> (T, 12, 3)
        skeleton_data = skeleton_data[:, Config.SELECTED_JOINTS, :]

        # -------------------------------------------------------------------------
        # 2. Construct Frame-wise Labels
        # -------------------------------------------------------------------------
        # Initialize with Background (0)
        labels_framewise = np.zeros(num_frames, dtype=np.int64)

        # Only process labels if they exist (Train/Val)
        raw_labels = getattr(video, "Labels", [])

        # Helper to process a single label entry
        def process_label_entry(lbl):
            try:
                name = lbl.Name
                start = int(lbl.Begin) - 1  # 1-based to 0-based
                end = int(
                    lbl.End
                )  # Exclusive in python slice? No, usually inclusive in Matlab.

                if name in Config.GESTURE_MAP:
                    gid = Config.GESTURE_MAP[name]
                    # Clamp to video boundaries
                    start = max(0, start)
                    end = min(num_frames, end)
                    if start < end:
                        labels_framewise[start:end] = gid
            except AttributeError:
                pass

        if isinstance(raw_labels, np.ndarray):
            if raw_labels.ndim == 0:
                process_label_entry(raw_labels.item())
            else:
                for l in raw_labels:
                    process_label_entry(l)
        else:
            process_label_entry(raw_labels)

    except Exception as e:
        # print(f"Error loading MAT {data_path}: {e}")
        return None

    # -------------------------------------------------------------------------
    # 3. Load and Process Audio
    # -------------------------------------------------------------------------
    try:
        # Load audio using torchaudio
        # waveform: (Channels, Time)
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if necessary (average across channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract MFCCs
        # Config.AUDIO_INPUT_SIZE is 13
        # We use torchaudio.transforms.MFCC
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=Config.AUDIO_INPUT_SIZE,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        )

        mfcc = mfcc_transform(waveform)  # (1, n_mfcc, Time)
        mfcc = mfcc.squeeze(0).numpy()  # (n_mfcc, Time)

        # Resample to match video frames
        # Axis 1 is time
        mfcc = scipy.signal.resample(mfcc, num_frames, axis=1)

        # Transpose to (T, n_mfcc)
        audio_data = mfcc.T.astype(np.float32)

    except Exception as e:
        # print(f"Error loading Audio {audio_path}: {e}")
        # Fallback: Zeros
        audio_data = np.zeros((num_frames, Config.AUDIO_INPUT_SIZE), dtype=np.float32)

    return {
        "skeleton": skeleton_data,  # (T, 12, 3)
        "audio": audio_data,  # (T, 13)
        "labels": labels_framewise,  # (T,)
    }


def physically_consistent_augmentation(positions):
    """
    Applies physically consistent augmentation:
    1. Perturb Raw Positions
    2. Derive Velocity from Perturbed Positions
    3. Normalize Positions (HipCenter)

    Args:
        positions (np.ndarray): Shape (T, J, 3)

    Returns:
        tuple: (augmented_normalized_pos, augmented_velocity)
    """
    T, J, C = positions.shape

    # 1. Perturb Raw Positions
    # Gaussian Noise
    noise = np.random.normal(0, 0.01, size=positions.shape).astype(
        np.float32
    )  # Small noise in meters/mm? Data is mm.
    # Wait, description says mm. 0.01 mm is nothing.
    # Let's check data scale. Usually Kinect is mm.
    # If mm, noise should be ~10-50mm. If meters, 0.01-0.05m.
    # Assuming mm based on description "expressed in millimeters".
    noise = np.random.normal(0, 5.0, size=positions.shape).astype(
        np.float32
    )  # 5mm std dev

    # Random Scaling (Simulate different user sizes)
    scale = np.random.uniform(0.9, 1.1)

    pos_perturbed = (positions + noise) * scale

    # 2. Derive Velocity from Perturbed Positions
    # V_t = P_t - P_{t-1}
    # Pad first frame with 0 velocity
    velocity = np.zeros_like(pos_perturbed)
    velocity[1:] = pos_perturbed[1:] - pos_perturbed[:-1]

    # 3. Normalize Positions
    # Subtract HipCenter (Joint 0) from all joints for each frame
    # Config.SELECTED_JOINTS[0] is HipCenter (index 0 in our subset)
    hip_center = pos_perturbed[:, 0:1, :]  # (T, 1, 3)
    pos_norm = pos_perturbed - hip_center

    return pos_norm, velocity


class SkeletonAudioDataset(Dataset):
    def __init__(
        self, metadata_csv, input_dir, cache_dir, split="train", load_cached_data=True
    ):
        """
        Args:
            metadata_csv (str): Path to metadata CSV.
            input_dir (str): Input directory containing raw files.
            cache_dir (str): Directory to store/load cached .npz files.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.split = split
        self.is_train = split == "train"
        self.df = pd.read_csv(metadata_csv)
        # Drop rows with missing paths to prevent TypeErrors (Cite debug_lesson_15)
        self.df = self.df.dropna(subset=["data_path", "audio_path"])
        self.input_dir = input_dir

        # Caching logic
        self.cache_path = os.path.join(cache_dir, f"{split}_data.npz")
        self.data_cache = []

        loaded = False
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading {split} data from cache: {self.cache_path}")
                # Load npz
                with np.load(self.cache_path, allow_pickle=True) as data:
                    # We stored as 'sample_0', 'sample_1'...
                    # Extract keys in order
                    keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
                    for k in keys:
                        self.data_cache.append(data[k].item())
                loaded = True
                print(f"Loaded {len(self.data_cache)} samples.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing.")
                loaded = False

        if not loaded:
            print(f"Processing {split} data from scratch...")
            processed_data = {}

            for idx, row in self.df.iterrows():
                sample = load_multimodal_data(row, input_dir)
                if sample is not None:
                    self.data_cache.append(sample)
                    processed_data[f"sample_{len(self.data_cache)-1}"] = sample

            # Save to cache
            print(f"Saving {split} data to cache: {self.cache_path}")
            np.savez_compressed(self.cache_path, **processed_data)

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        sample = self.data_cache[idx]

        # Extract raw data
        # Copy to avoid modifying cache during augmentation
        skeleton = sample["skeleton"].copy()  # (T, 12, 3)
        audio = sample["audio"].copy()  # (T, 13)
        labels = sample["labels"].copy()  # (T,)

        # Preprocessing / Augmentation
        if self.is_train:
            # Apply Physically Consistent Augmentation
            pos, vel = physically_consistent_augmentation(skeleton)
        else:
            # Deterministic Normalization for Val/Test
            # 1. No perturbation
            # 2. Derive Velocity
            velocity = np.zeros_like(skeleton)
            velocity[1:] = skeleton[1:] - skeleton[:-1]

            # 3. Normalize (HipCenter at index 0)
            hip_center = skeleton[:, 0:1, :]
            pos = skeleton - hip_center
            vel = velocity

        # Flatten Joint features for input to model
        # Input: (T, J, 3) -> (T, J*3)
        T = pos.shape[0]
        pos_flat = pos.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)

        # Convert to Torch Tensors
        pos_tensor = torch.from_numpy(pos_flat).float()
        vel_tensor = torch.from_numpy(vel_flat).float()
        audio_tensor = torch.from_numpy(audio).float()
        labels_tensor = torch.from_numpy(labels).long()

        return {
            "pos": pos_tensor,
            "vel": vel_tensor,
            "audio": audio_tensor,
            "labels": labels_tensor,
            "length": T,
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    """
    # Sort by length (descending) for pack_padded_sequence if needed (optional but good practice)
    batch.sort(key=lambda x: x["length"], reverse=True)

    lengths = torch.tensor([x["length"] for x in batch])
    max_len = lengths.max().item()

    # Feature sizes
    pos_dim = batch[0]["pos"].shape[1]
    vel_dim = batch[0]["vel"].shape[1]
    aud_dim = batch[0]["audio"].shape[1]

    # Pre-allocate padded tensors
    batch_size = len(batch)
    padded_pos = torch.zeros(batch_size, max_len, pos_dim)
    padded_vel = torch.zeros(batch_size, max_len, vel_dim)
    padded_aud = torch.zeros(batch_size, max_len, aud_dim)
    padded_labels = torch.zeros(
        batch_size, max_len, dtype=torch.long
    )  # 0 is background

    for i, sample in enumerate(batch):
        L = sample["length"]
        padded_pos[i, :L, :] = sample["pos"]
        padded_vel[i, :L, :] = sample["vel"]
        padded_aud[i, :L, :] = sample["audio"]
        padded_labels[i, :L] = sample["labels"]

    return {
        "pos": padded_pos,  # (B, T, D_pos)
        "vel": padded_vel,  # (B, T, D_vel)
        "audio": padded_aud,  # (B, T, D_aud)
        "labels": padded_labels,  # (B, T)
        "lengths": lengths,  # (B,)
    }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Train Set
    train_ds = SkeletonAudioDataset(
        metadata_csv=Config.TRAIN_CSV,
        input_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        split="train",
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Val Set
    val_ds = SkeletonAudioDataset(
        metadata_csv=Config.VAL_CSV,
        input_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        split="val",
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test Set
    test_ds = SkeletonAudioDataset(
        metadata_csv=Config.TEST_CSV,
        input_dir=Config.INPUT_DIR,
        cache_dir=Config.CACHE_DIR,
        split="test",
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
