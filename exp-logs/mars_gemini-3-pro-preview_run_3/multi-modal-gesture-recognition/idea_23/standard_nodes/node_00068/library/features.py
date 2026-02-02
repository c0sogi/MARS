import os
import numpy as np
import pandas as pd
from library import config, data_utils

# ==========================================
# 1. Kinematic Feature Extraction & Augmentation
# ==========================================


def augment_skeleton(skeleton):
    """
    Applies random 3D rotation (Y-axis) and scaling to the skeleton sequence.
    This is applied to raw positions to ensure subsequent derivatives are consistent.

    Args:
        skeleton: np.ndarray of shape (T, 20, 3)

    Returns:
        augmented_skeleton: np.ndarray of shape (T, 20, 3)
    """
    # Copy to avoid modifying original data
    skel_aug = skeleton.copy()

    # Random Rotation around Y-axis (-15 to +15 degrees)
    # theta in radians
    theta = np.random.uniform(-15, 15) * (np.pi / 180.0)

    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix for Y-axis
    rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    # Apply rotation: (T, 20, 3) -> reshape to (T*20, 3) -> matmul -> reshape back
    T, J, C = skel_aug.shape
    flat_skel = skel_aug.reshape(-1, 3)
    flat_skel = np.dot(flat_skel, rotation_matrix.T)
    skel_aug = flat_skel.reshape(T, J, C)

    # Random Scaling (0.85 to 1.15)
    scale = np.random.uniform(0.85, 1.15)
    skel_aug = skel_aug * scale

    return skel_aug


def extract_kinematics(skeleton, augment=False):
    """
    Computes Position, Velocity, and Acceleration from skeleton data.
    Ensures kinematic consistency by augmenting *before* derivation.

    Args:
        skeleton: np.ndarray (T, 20, 3)
        augment: bool, whether to apply augmentation

    Returns:
        features: np.ndarray (T, 180) -> [Pos, Vel, Acc]
    """
    # 1. Augmentation (if enabled)
    if augment:
        skel_proc = augment_skeleton(skeleton)
    else:
        skel_proc = skeleton.copy()

    # 2. Flatten Joints: (T, 20, 3) -> (T, 60)
    T = skel_proc.shape[0]
    position = skel_proc.reshape(T, -1)

    # 3. Compute Derivatives (Central Difference)
    # Velocity: dP/dt
    # np.gradient handles boundaries automatically
    velocity = np.gradient(position, axis=0)

    # Acceleration: dV/dt
    acceleration = np.gradient(velocity, axis=0)

    # 4. Concatenate
    # Shape: (T, 60+60+60) = (T, 180)
    features = np.concatenate([position, velocity, acceleration], axis=1)

    return features.astype(np.float32)


# ==========================================
# 2. Sample Processing & Fusion
# ==========================================


def process_sample(skeleton, audio, stats=None, augment=False):
    """
    Fuses skeleton kinematics and audio features into a single input vector.
    Applies normalization if stats are provided.

    Args:
        skeleton: (T, 20, 3)
        audio: (T, 13)
        stats: dict containing 'mean' and 'std' (optional)
        augment: bool

    Returns:
        fused_features: (T, 193)
    """
    # 1. Extract Kinematics (T, 180)
    kinematics = extract_kinematics(skeleton, augment=augment)

    # 2. Align lengths
    # Audio and Skeleton should be aligned by data_utils, but we take the min length for safety
    min_len = min(kinematics.shape[0], audio.shape[0])
    kinematics = kinematics[:min_len]
    audio_feat = audio[:min_len]

    # 3. Normalization
    if stats is not None:
        # Normalize Kinematics
        k_mean = stats["kinematics_mean"]
        k_std = stats["kinematics_std"]
        # Avoid division by zero
        k_std = np.where(k_std < 1e-6, 1.0, k_std)

        kinematics = (kinematics - k_mean) / k_std

        # Normalize Audio
        if "audio_mean" in stats:
            a_mean = stats["audio_mean"]
            a_std = stats["audio_std"]
            a_std = np.where(a_std < 1e-6, 1.0, a_std)
            audio_feat = (audio_feat - a_mean) / a_std

    # 4. Early Fusion (Concatenation)
    # (T, 180) + (T, 13) -> (T, 193)
    fused = np.concatenate([kinematics, audio_feat], axis=1)

    return fused.astype(np.float32)


# ==========================================
# 3. Data Loading & Stats Caching
# ==========================================


def compute_stats(skeletons, audios):
    """
    Computes global mean and std for kinematics and audio over the entire dataset.
    Used for standardization (Z-score normalization).
    """
    print("Computing normalization statistics from training set...")

    # Accumulators
    k_sum = np.zeros(config.SKELETON_INPUT_DIM, dtype=np.float64)
    k_sq_sum = np.zeros(config.SKELETON_INPUT_DIM, dtype=np.float64)
    k_count = 0

    a_sum = np.zeros(config.AUDIO_INPUT_DIM, dtype=np.float64)
    a_sq_sum = np.zeros(config.AUDIO_INPUT_DIM, dtype=np.float64)
    a_count = 0

    for skel, aud in zip(skeletons, audios):
        # Compute kinematics (No augmentation for stats calculation)
        feat = extract_kinematics(skel, augment=False)

        # Accumulate Kinematics
        k_sum += np.sum(feat, axis=0)
        k_sq_sum += np.sum(feat**2, axis=0)
        k_count += feat.shape[0]

        # Accumulate Audio
        a_sum += np.sum(aud, axis=0)
        a_sq_sum += np.sum(aud**2, axis=0)
        a_count += aud.shape[0]

    # Finalize Kinematics
    k_mean = (k_sum / k_count).astype(np.float32)
    k_std = np.sqrt((k_sq_sum / k_count) - (k_mean**2)).astype(np.float32)

    # Finalize Audio
    a_mean = (a_sum / a_count).astype(np.float32)
    a_std = np.sqrt((a_sq_sum / a_count) - (a_mean**2)).astype(np.float32)

    return {
        "kinematics_mean": k_mean,
        "kinematics_std": k_std,
        "audio_mean": a_mean,
        "audio_std": a_std,
    }


def load_data_and_stats(load_cached_data=True):
    """
    Loads train, val, and test datasets using data_utils.
    Computes or loads normalization statistics.

    Args:
        load_cached_data: bool, if True, attempts to load from disk.

    Returns:
        train_data: dict (skeleton, audio, labels, sample_ids)
        val_data: dict
        test_data: dict
        stats: dict (normalization parameters)
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Process Datasets (Raw Loading)
    # data_utils handles caching of the raw data structures
    train_data = data_utils.process_dataset(df_train, "train", load_cached_data)
    val_data = data_utils.process_dataset(df_val, "val", load_cached_data)
    test_data = data_utils.process_dataset(df_test, "test", load_cached_data)

    # 3. Handle Statistics (Caching logic)
    stats_path = os.path.join(config.CACHE_DIR, "stats.npz")

    stats = None
    if load_cached_data and os.path.exists(stats_path):
        try:
            print(f"Loading statistics from {stats_path}...")
            loaded = np.load(stats_path)
            stats = {k: loaded[k] for k in loaded.files}
        except Exception as e:
            print(f"Failed to load stats: {e}")

    if stats is None:
        # Compute from training data
        stats = compute_stats(train_data["skeleton"], train_data["audio"])

        # Save
        print(f"Saving statistics to {stats_path}...")
        np.savez(stats_path, **stats)

    return train_data, val_data, test_data, stats
