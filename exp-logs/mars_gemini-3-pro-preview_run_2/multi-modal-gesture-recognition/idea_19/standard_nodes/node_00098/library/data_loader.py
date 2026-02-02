import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import interpolate

from library.config import Config
from library.utils import set_seed


def load_and_process_data(split_name, load_cached_data=True):
    """
    Loads and processes data for a specific split (train, val, test).
    Uses caching to speed up subsequent runs.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Reconstruct list of arrays from the npz archive
            # We store them as object arrays or individual keys.
            # For simplicity, we assume they are stored as object arrays under specific keys.
            skeletons = list(loaded["skeletons"])
            audio_features = list(loaded["audio_features"])
            labels = list(loaded["labels"])
            boundaries = list(loaded["boundaries"])
            sample_ids = list(loaded["sample_ids"])
            return skeletons, audio_features, labels, boundaries, sample_ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from scratch...")
    csv_path = os.path.join(Config.METADATA_DIR, f"{split_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    skeletons_list = []
    audio_list = []
    labels_list = []
    boundaries_list = []
    sample_ids_list = []

    # MFCC Transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=Config.AUDIO_MFCC_N_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
        audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

        # --- Load MAT File (Skeleton & Labels) ---
        try:
            mat = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
            video = mat["Video"]
            num_frames = int(video.NumFrames)

            # Extract Skeleton
            # Video.Frames is typically an array of structs
            frames = video.Frames

            # Pre-allocate skeleton array: (T, Joints, 3)
            skel_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

            # Robust parsing of frames
            # Sometimes frames is a single object if NumFrames=1, but usually an array
            if not isinstance(frames, np.ndarray) and num_frames > 1:
                frames = [frames]  # Should not happen with squeeze_me=True usually
            elif not isinstance(frames, np.ndarray) and num_frames == 1:
                frames = np.array([frames])

            for t in range(min(num_frames, len(frames))):
                frame_obj = frames[t]
                # Skeleton might be an array (multiple users) or single struct
                # We assume the first skeleton is the target user
                try:
                    skel_obj = frame_obj.Skeleton
                    if isinstance(skel_obj, np.ndarray) and len(skel_obj) > 0:
                        curr_skel = skel_obj[0]  # Take first user
                    else:
                        curr_skel = skel_obj

                    # Extract specific joints
                    # Assuming JointsType is not easily mappable, we rely on the fact that
                    # WorldPosition is available.
                    # IMPORTANT: The dataset description implies a struct.
                    # However, standard Kinect MAT files usually have skeletons as indexable arrays
                    # if we know the index.
                    # If curr_skel is a struct with fields like 'HipCenter', we use getattr.
                    # If it is an array of joints, we use index.

                    # Let's try to detect structure.
                    # Based on description: "JointsType can be as follows: HipCenter..."
                    # This implies fields.

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
                    ]

                    for j_idx, j_enum in enumerate(Config.SKELETON_JOINTS):
                        # Map index to name if possible, or try direct indexing if it's an array
                        # The Config.SKELETON_JOINTS are indices 0..11 corresponding to the list above.
                        j_name = joint_names[j_enum]

                        # Try getting by name
                        if hasattr(curr_skel, j_name):
                            joint_node = getattr(curr_skel, j_name)
                        # Try getting by index (if curr_skel is array of joints)
                        elif (
                            isinstance(curr_skel, (list, np.ndarray))
                            and len(curr_skel) > j_enum
                        ):
                            joint_node = curr_skel[j_enum]
                        else:
                            # Fallback: try getting 'Joint' array
                            if (
                                hasattr(curr_skel, "Joint")
                                and len(curr_skel.Joint) > j_enum
                            ):
                                joint_node = curr_skel.Joint[j_enum]
                            else:
                                # Zero if not found
                                continue

                        if hasattr(joint_node, "WorldPosition"):
                            wp = joint_node.WorldPosition
                            skel_data[t, j_idx, 0] = wp.X
                            skel_data[t, j_idx, 1] = wp.Y
                            skel_data[t, j_idx, 2] = wp.Z

                except Exception:
                    # If frame parsing fails, keep zeros (padding/missing)
                    pass

            # --- Construct Targets ---
            # Labels: (T,) class indices
            # Boundaries: (T,) 1.0 if transition, 0.0 otherwise
            target_labels = np.zeros(num_frames, dtype=np.int64)  # 0 is background
            target_boundaries = np.zeros(num_frames, dtype=np.float32)

            # Parse labels from MAT (Ground Truth)
            # Only available for train/val (not empty)
            raw_labels = getattr(video, "Labels", [])

            # Helper to process single label entry
            def process_label_entry(entry):
                try:
                    name = entry.Name
                    start = int(entry.Begin) - 1  # 1-based to 0-based
                    end = int(entry.End) - 1
                    if name in Config.GESTURE_MAP:
                        gid = Config.GESTURE_MAP[name]
                        # Clip to valid range
                        start = max(0, start)
                        end = min(num_frames - 1, end)
                        if start <= end:
                            target_labels[start : end + 1] = gid
                            # Mark boundaries
                            target_boundaries[start] = 1.0
                            if end + 1 < num_frames:
                                target_boundaries[end + 1] = 1.0
                except AttributeError:
                    pass

            if isinstance(raw_labels, np.ndarray):
                if raw_labels.ndim == 0 and raw_labels.size > 0:
                    process_label_entry(raw_labels.item())
                else:
                    for l in raw_labels:
                        process_label_entry(l)
            elif raw_labels:  # Single object
                process_label_entry(raw_labels)

            # --- Load Audio ---
            if os.path.exists(audio_path) and Config.USE_AUDIO:
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    # Resample if needed
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)

                    # Compute MFCC: (1, n_mfcc, time)
                    mfcc = mfcc_transform(waveform)
                    mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

                    # Interpolate to match video frames: (n_mfcc, num_frames)
                    # Add batch dim for interpolate
                    mfcc = mfcc.unsqueeze(0)
                    mfcc = interpolate(
                        mfcc, size=num_frames, mode="linear", align_corners=False
                    )
                    mfcc = mfcc.squeeze(0).permute(1, 0)  # (num_frames, n_mfcc)

                    audio_data = mfcc.numpy()
                except Exception:
                    audio_data = np.zeros(
                        (num_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32
                    )
            else:
                audio_data = np.zeros(
                    (num_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32
                )

            # Append
            skeletons_list.append(skel_data)
            audio_list.append(audio_data)
            labels_list.append(target_labels)
            boundaries_list.append(target_boundaries)
            sample_ids_list.append(sample_id)

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue

    # 3. Save to Cache
    # Use object array to store variable length sequences
    skeletons_arr = np.array(skeletons_list, dtype=object)
    audio_arr = np.array(audio_list, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)
    boundaries_arr = np.array(boundaries_list, dtype=object)
    sample_ids_arr = np.array(sample_ids_list, dtype=object)

    np.savez_compressed(
        cache_path,
        skeletons=skeletons_arr,
        audio_features=audio_arr,
        labels=labels_arr,
        boundaries=boundaries_arr,
        sample_ids=sample_ids_arr,
    )

    return skeletons_list, audio_list, labels_list, boundaries_list, sample_ids_list


class GestureDataset(Dataset):
    def __init__(self, skeletons, audio_features, labels, boundaries, is_train=True):
        self.skeletons = skeletons
        self.audio_features = audio_features
        self.labels = labels
        self.boundaries = boundaries
        self.is_train = is_train

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        # Load raw data
        # skel: (T, J, 3)
        skel = self.skeletons[idx].copy()
        audio = self.audio_features[idx].copy()
        label = self.labels[idx].copy()
        boundary = self.boundaries[idx].copy()

        T, J, C = skel.shape

        # 1. Augmentation (Train only)
        # Physically consistent noise: Smooth noise added to position BEFORE velocity calc
        if self.is_train:
            # Generate Gaussian noise
            sigma = 10.0  # 10mm noise
            noise = np.random.normal(0, sigma, skel.shape)
            # Apply temporal smoothing (Low-pass filter)
            # Sigma=1.0 along time axis (axis 0)
            noise = scipy.ndimage.gaussian_filter1d(noise, sigma=1.0, axis=0)
            skel = skel + noise

        # 2. Normalization
        # Centering: Subtract HipCenter (Joint 0) from all joints
        # skel shape (T, J, 3). HipCenter is skel[:, 0, :] -> (T, 3)
        hip_center = skel[:, 0:1, :]  # Keep dims for broadcasting
        skel = skel - hip_center

        # Scaling: mm to meters
        skel = skel * Config.SCALE_FACTOR

        # 3. Feature Engineering
        # Velocity: Temporal difference
        # Pad first frame with zero velocity
        vel = np.zeros_like(skel)
        vel[1:] = skel[1:] - skel[:-1]

        # Flatten Skeleton and Velocity: (T, J*3)
        skel_flat = skel.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)

        # Concatenate: [Skeleton, Velocity, Audio]
        # Audio is (T, n_mfcc)
        features = np.concatenate([skel_flat, vel_flat, audio], axis=1)

        # Convert to Tensor
        features_t = torch.tensor(features, dtype=torch.float32)
        labels_t = torch.tensor(label, dtype=torch.long)
        boundaries_t = torch.tensor(boundary, dtype=torch.float32)

        return features_t, labels_t, boundaries_t


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Returns:
        padded_features: (B, T_max, F)
        padded_labels: (B, T_max)
        padded_boundaries: (B, T_max)
        mask: (B, T_max) - 1 for valid frames, 0 for padding
    """
    features, labels, boundaries = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(f) for f in features])

    # Pad sequences (batch_first=True)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=0
    )  # 0 is background
    padded_boundaries = pad_sequence(boundaries, batch_first=True, padding_value=0.0)

    # Create Mask
    max_len = padded_features.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()  # (B, T)

    return padded_features, padded_labels, padded_boundaries, mask


def get_data_loaders(load_cached_data=True):
    """
    Initializes datasets and dataloaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)

    # Load Data
    train_data = load_and_process_data("train", load_cached_data)
    val_data = load_and_process_data("val", load_cached_data)
    test_data = load_and_process_data("test", load_cached_data)

    # Create Datasets
    # train_data tuple: (skeletons, audio, labels, boundaries, ids)

    # Debug Subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        print(f"Debugging with subset size: {Config.DEBUG_SUBSET_SIZE}")
        train_ds = GestureDataset(
            *[d[: Config.DEBUG_SUBSET_SIZE] for d in train_data], is_train=True
        )
        val_ds = GestureDataset(
            *[d[: Config.DEBUG_SUBSET_SIZE] for d in val_data], is_train=False
        )
    else:
        train_ds = GestureDataset(*train_data[:4], is_train=True)
        val_ds = GestureDataset(*val_data[:4], is_train=False)

    test_ds = GestureDataset(*test_data[:4], is_train=False)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_data[4]  # Return test IDs
