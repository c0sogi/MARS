import os
import json
import random
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
JOINT_NAMES = [
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

JOINT_MAP = {name: i for i, name in enumerate(JOINT_NAMES)}

# ==========================================
# Helper Functions
# ==========================================


def polymorphic_mat_parse(mat_path):
    """
    Robustly parses the .mat file to extract skeleton data.
    Handles variations in struct arrays vs cell arrays.
    Returns: numpy array of shape (NumFrames, 20, 3) or None if failed.
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None
        video = mat["Video"]

        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames
        num_frames = len(frames) if isinstance(frames, (list, np.ndarray)) else 1

        # Pre-allocate
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        # Ensure frames is iterable
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        for f_idx, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton
            # Handle case where Skeleton might be an array (multiple users) or single struct
            if isinstance(skel, (list, np.ndarray)) and len(skel) > 0:
                # Heuristic: Take the first tracked skeleton
                skel = skel[0]

            # Now we expect skel to have JointsType and WorldPosition
            # Check if it's a struct with fields or array of structs
            # Based on description: "An array of Skeleton structures... JointsType... WorldPosition"
            # It seems 'skel' itself might be the array of joints?
            # Or 'skel' is the user, containing joints?
            # Let's try to find joint data.

            # Case A: skel is an object with 'JointsType' and 'WorldPosition' which are arrays
            # Case B: skel is an array of objects, each being a joint

            joints_found = 0

            # Flatten if necessary
            if isinstance(skel, (list, np.ndarray)):
                joint_list = skel
            else:
                # If it's a single object, maybe it has properties that are arrays?
                # But usually in these datasets, it's an array of structs.
                # Let's assume it's iterable or try to access fields.
                try:
                    joint_list = [skel]  # Unlikely to be just 1 joint
                except:
                    continue

            for joint in joint_list:
                if not hasattr(joint, "JointsType") or not hasattr(
                    joint, "WorldPosition"
                ):
                    continue

                j_type = str(joint.JointsType)
                if j_type in JOINT_MAP:
                    j_idx = JOINT_MAP[j_type]
                    pos = joint.WorldPosition
                    # pos should have X, Y, Z
                    if hasattr(pos, "X") and hasattr(pos, "Y") and hasattr(pos, "Z"):
                        skeleton_data[f_idx, j_idx, 0] = pos.X
                        skeleton_data[f_idx, j_idx, 1] = pos.Y
                        skeleton_data[f_idx, j_idx, 2] = pos.Z
                        joints_found += 1

            # If no joints found via iteration, maybe structure is different (Struct of Arrays)
            if joints_found == 0:
                pass  # Fallback or error handling could go here

        return skeleton_data

    except Exception as e:
        print(f"Error parsing {mat_path}: {e}")
        return None


def extract_audio_features(wav_path, target_num_frames):
    """
    Loads audio, computes MFCC, and resamples to match video frame count.
    """
    try:
        waveform, sample_rate = torchaudio.load(wav_path)

        # Compute MFCC
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)  # [Channels, n_mfcc, time]

        # Average over audio channels if stereo
        if mfcc.shape[0] > 1:
            mfcc = torch.mean(mfcc, dim=0, keepdim=True)

        # mfcc shape: [1, n_mfcc, time]
        # Resize to match video frames
        # We treat MFCCs as a 1D signal with C channels and interpolate time
        mfcc = torch.nn.functional.interpolate(
            mfcc.unsqueeze(0),  # [1, 1, n_mfcc, time]
            size=(Config.N_MFCC, target_num_frames),
            mode="bilinear",
            align_corners=False,
        )

        # Shape: [1, 1, n_mfcc, target_num_frames] -> [target_num_frames, n_mfcc]
        mfcc = mfcc.squeeze().permute(1, 0).numpy()
        return mfcc

    except Exception as e:
        # Return zeros if audio fails
        return np.zeros((target_num_frames, Config.N_MFCC), dtype=np.float32)


def rotate_skeleton(skeleton_data):
    """
    Applies random Y-axis rotation to the skeleton data.
    skeleton_data: (T, J, 3)
    """
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    # Apply rotation: (T, J, 3) @ (3, 3) -> (T, J, 3)
    # Reshape to (T*J, 3) for matmul
    T, J, C = skeleton_data.shape
    flat_skel = skeleton_data.reshape(-1, 3)
    rotated = np.dot(flat_skel, rotation_matrix.T)
    return rotated.reshape(T, J, C)


def process_sequence(sample, is_train=True):
    """
    Loads and processes a single sequence.
    Returns: (features, labels)
    """
    # Paths
    mat_path = os.path.join(Config.INPUT_DIR, sample["data_path"])
    audio_path = os.path.join(Config.INPUT_DIR, sample["audio_path"])

    # 1. Load Skeleton
    skeleton = polymorphic_mat_parse(mat_path)
    if skeleton is None:
        return None, None

    num_frames = skeleton.shape[0]
    if num_frames < Config.MIN_GESTURE_LENGTH:
        return None, None

    # 2. Root-Relative Centering
    # HipCenter is index 0 based on JOINT_NAMES
    hip_pos = skeleton[:, 0:1, :]  # (T, 1, 3)
    skeleton = skeleton - hip_pos

    # 3. Augmentation 1: Random Rotation (Geometric)
    # Only apply during training generation to create a varied cached dataset
    if is_train:
        skeleton = rotate_skeleton(skeleton)

    # 4. Augmentation 2: Noise Injection (Before Derivation)
    # Inject Gaussian noise to positions
    noise = np.random.normal(0, Config.NOISE_SIGMA, skeleton.shape)
    skeleton_noisy = skeleton + noise

    # 5. Kinematic Derivation (Vel, Acc)
    # Gradient across time (axis 0)
    velocity = np.gradient(skeleton_noisy, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    # Stack: (T, J, 3) -> (T, J, 9)
    # Concatenate along the last dimension
    kinematic_features = np.concatenate(
        [skeleton_noisy, velocity, acceleration], axis=2
    )
    # Flatten joints: (T, J*9) -> (T, 180)
    kinematic_features = kinematic_features.reshape(num_frames, -1)

    # 6. Audio Features
    audio_features = extract_audio_features(audio_path, num_frames)

    # 7. Early Fusion
    # (T, 180) + (T, 13) -> (T, 193)
    features = np.concatenate([kinematic_features, audio_features], axis=1)

    # 8. Labels
    labels = np.zeros(num_frames, dtype=np.int64)
    if is_train:
        label_list = json.loads(sample["labels"])
        for l in label_list:
            start = max(0, l["begin"] - 1)  # 1-based to 0-based
            end = min(num_frames, l["end"])
            labels[start:end] = l["id"]

    return features.astype(np.float32), labels


def create_windows(features, labels, window_size, stride, is_train=True):
    """
    Slices sequence into windows.
    """
    num_frames = features.shape[0]
    windows_feat = []
    windows_label = []

    if num_frames < window_size:
        # Pad if shorter than window
        pad_len = window_size - num_frames
        feat_pad = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
        label_pad = np.pad(labels, (0, pad_len), mode="constant")
        return [feat_pad], [label_pad]

    step = stride if is_train else window_size // 2  # Overlap for test too

    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size
        windows_feat.append(features[start:end])
        windows_label.append(labels[start:end])

    # Handle remainder for test/val to ensure full coverage
    if not is_train and (num_frames - window_size) % step != 0:
        start = num_frames - window_size
        windows_feat.append(features[start:])
        windows_label.append(labels[start:])

    return windows_feat, windows_label


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, data_dict):
        self.features = data_dict["features"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: [Time, Dim] -> Transpose to [Dim, Time] for 1D Conv/TCN
        x = torch.from_numpy(self.features[idx]).float().transpose(0, 1)
        y = torch.from_numpy(self.labels[idx]).long()
        return x, y


# ==========================================
# Data Processing & Caching
# ==========================================


def process_and_cache_data(metadata_path, cache_path, mode="train", debug=False):
    """
    Processes raw data, creates windows, and caches to .npz.
    """
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    all_features = []
    all_labels = []

    is_train_mode = mode == "train"

    for idx, row in df.iterrows():
        feat, lbl = process_sequence(row, is_train=is_train_mode)
        if feat is None:
            continue

        # Create windows
        # For test set, we might want full sequences or windows.
        # The prompt implies window-based training.
        # For inference, we usually process full sequences, but to fit the model architecture (TCN),
        # fixed windows are easier. However, the prompt says "Full Sequence Inference".
        # If we use batch size 1, we can have variable length.
        # But the TCN/GRU might expect fixed size if we batch.
        # Let's stick to windows for training, and maybe windows or full seq for val/test.
        # To be safe and consistent with "Stochastic-Depth" training on windows:

        w_feats, w_lbls = create_windows(
            feat, lbl, Config.WINDOW_SIZE, Config.STRIDE, is_train=is_train_mode
        )

        all_features.extend(w_feats)
        all_labels.extend(w_lbls)

    # Stack
    # Note: If variable length (e.g. padding edge cases), this might fail.
    # create_windows ensures fixed size padding or slicing.

    features_np = np.array(all_features, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.int64)

    print(f"Saving cache to {cache_path} (Shape: {features_np.shape})")
    np.savez_compressed(cache_path, features=features_np, labels=labels_np)

    return {"features": features_np, "labels": labels_np}


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles caching logic.
    """
    # Ensure working directory
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "dataset_train.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "dataset_val.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "dataset_test.npz")

    loaders = {}

    # --- Train ---
    if load_cached_data and os.path.exists(train_cache):
        print("Loading Train Cache...")
        train_data = np.load(train_cache)
        train_dict = {
            "features": train_data["features"],
            "labels": train_data["labels"],
        }
    else:
        train_dict = process_and_cache_data(
            Config.TRAIN_METADATA_PATH, train_cache, mode="train", debug=Config.DEBUG
        )

    train_dataset = GestureDataset(train_dict)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Val ---
    if load_cached_data and os.path.exists(val_cache):
        print("Loading Val Cache...")
        val_data = np.load(val_cache)
        val_dict = {"features": val_data["features"], "labels": val_data["labels"]}
    else:
        val_dict = process_and_cache_data(
            Config.VAL_METADATA_PATH, val_cache, mode="val", debug=Config.DEBUG
        )

    val_dataset = GestureDataset(val_dict)
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    # Note: Test set has no labels, but we keep structure consistent
    if load_cached_data and os.path.exists(test_cache):
        print("Loading Test Cache...")
        test_data = np.load(test_cache)
        test_dict = {"features": test_data["features"], "labels": test_data["labels"]}
    else:
        test_dict = process_and_cache_data(
            Config.TEST_METADATA_PATH, test_cache, mode="test", debug=Config.DEBUG
        )

    test_dataset = GestureDataset(test_dict)
    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loaders
