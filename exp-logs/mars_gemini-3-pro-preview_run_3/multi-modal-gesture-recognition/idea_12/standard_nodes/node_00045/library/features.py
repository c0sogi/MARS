import numpy as np
import os
from library.config import Config

# Set random seed for reproducibility
np.random.seed(Config.SEED)

# Parent indices for Kinect v1/v2 20-joint skeleton
# Maps index i to its parent index. -1 indicates root.
# Based on JOINT_MAP in data_utils.py:
# 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head, 4:ShoulderLeft...
PARENTS = np.array(
    [
        -1,  # 0: HipCenter (Root)
        0,  # 1: Spine
        1,  # 2: ShoulderCenter
        2,  # 3: Head
        2,  # 4: ShoulderLeft
        4,  # 5: ElbowLeft
        5,  # 6: WristLeft
        6,  # 7: HandLeft
        2,  # 8: ShoulderRight
        8,  # 9: ElbowRight
        9,  # 10: WristRight
        10,  # 11: HandRight
        0,  # 12: HipLeft
        12,  # 13: KneeLeft
        13,  # 14: AnkleLeft
        14,  # 15: FootLeft
        0,  # 16: HipRight
        16,  # 17: KneeRight
        17,  # 18: AnkleRight
        18,  # 19: FootRight
    ]
)


class FeatureNormalizer:
    """
    Handles normalization of the feature vectors using standardization (mean=0, std=1).
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.epsilon = 1e-8

    def fit(self, features_list):
        """
        Compute mean and std from a list of feature arrays.
        Args:
            features_list: List of np.ndarray (T, 240)
        """
        if not features_list:
            return

        # Concatenate all sequences to compute global stats
        all_data = np.concatenate(features_list, axis=0)
        self.mean = np.mean(all_data, axis=0).astype(np.float32)
        self.std = np.std(all_data, axis=0).astype(np.float32)

        # Avoid division by zero
        self.std[self.std < self.epsilon] = 1.0

    def transform(self, features):
        """
        Apply normalization.
        Args:
            features: np.ndarray (T, 240)
        Returns:
            np.ndarray (T, 240)
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before transform.")
        return (features - self.mean) / self.std

    def save(self, path):
        """Save normalization stats to .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """Load normalization stats from .npz file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalizer file not found: {path}")
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


def augment_skeleton(skeleton, rotation_range=30, scale_range=0.2):
    """
    Apply random rotation around Y-axis and scaling.
    Args:
        skeleton: (T, 20, 3)
        rotation_range: Max rotation in degrees.
        scale_range: Max deviation from 1.0.
    Returns:
        Augmented skeleton (T, 20, 3)
    """
    aug_skeleton = skeleton.copy()

    # 1. Rotation around Y-axis
    theta_deg = np.random.uniform(-rotation_range, rotation_range)
    theta = np.radians(theta_deg)

    c, s = np.cos(theta), np.sin(theta)

    # Apply rotation matrix manually for Y-axis
    # x' = x*c + z*s
    # z' = -x*s + z*c
    x = aug_skeleton[..., 0]
    z = aug_skeleton[..., 2]

    aug_skeleton[..., 0] = x * c + z * s
    aug_skeleton[..., 2] = -x * s + z * c

    # 2. Scaling
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    aug_skeleton *= scale

    return aug_skeleton


def compute_bone_vectors(skeleton):
    """
    Compute vectors from parent joint to child joint.
    Args:
        skeleton: (T, 20, 3)
    Returns:
        bones: (T, 20, 3)
    """
    bones = np.zeros_like(skeleton)

    for i, parent_idx in enumerate(PARENTS):
        if parent_idx == -1:
            # Root joint: bone vector is 0
            bones[:, i, :] = 0.0
        else:
            bones[:, i, :] = skeleton[:, i, :] - skeleton[:, parent_idx, :]

    return bones


def compute_kinematics(features):
    """
    Compute Velocity and Acceleration.
    Args:
        features: (T, 20, 3)
    Returns:
        velocity: (T, 20, 3)
        acceleration: (T, 20, 3)
    """
    # Velocity: P_t - P_{t-1}
    # Pad start with 0 (replicate first frame diff is 0)
    velocity = np.diff(features, axis=0, prepend=features[0:1])

    # Acceleration: V_t - V_{t-1}
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    return velocity, acceleration


def extract_features(skeleton, augment=False):
    """
    Main feature extraction pipeline implementing the SA-AKN strategy.

    Pipeline:
    1. Augment (if True)
    2. Compute Root-Relative Positions (P_rel = P - P_spine)
    3. Compute Bone Vectors (Spatial Structure)
    4. Compute Velocity (Temporal Dynamics)
    5. Compute Acceleration (Temporal Dynamics)
    6. Concatenate and Flatten

    Args:
        skeleton: (T, 20, 3) Raw World Coordinates
        augment: bool

    Returns:
        features: (T, 240) Flattened feature vector
    """
    # Ensure float32
    skeleton = skeleton.astype(np.float32)

    # 1. Augmentation
    if augment:
        skeleton = augment_skeleton(skeleton)

    # 2. Root-Relative Positions
    # Config says: "P_rel = P_aug - P_spine".
    # Index 1 is Spine.
    spine_pos = skeleton[:, 1:2, :]  # (T, 1, 3)
    rel_pos = skeleton - spine_pos  # (T, 20, 3)

    # 3. Bone Vectors
    # Calculated on relative positions (mathematically same as absolute for vectors)
    bones = compute_bone_vectors(rel_pos)  # (T, 20, 3)

    # 4. Kinematics (Velocity & Acceleration)
    # Calculated on Relative Positions
    vel, acc = compute_kinematics(rel_pos)  # (T, 20, 3), (T, 20, 3)

    # 5. Concatenation
    # Stack features: [RelativePos, Bones, Velocity, Acceleration]
    # Shape: (T, 20, 3+3+3+3) = (T, 20, 12)
    combined = np.concatenate([rel_pos, bones, vel, acc], axis=-1)

    # Flatten to (T, 240)
    T, J, C = combined.shape
    flattened = combined.reshape(T, J * C)

    return flattened
