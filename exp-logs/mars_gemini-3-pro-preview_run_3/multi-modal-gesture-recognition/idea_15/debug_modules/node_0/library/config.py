import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Robust Spatial-Kinematic Attentive Refinement Network (RSK-ARN).
    Handles paths, hyperparameters, and feature definitions.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 15 (Caching & Outputs)
    WORKING_DIR = "./working/idea_15"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    # Sliding Window Logic (Atomic gesture duration ~64 frames)
    WINDOW_SIZE = 64
    OVERLAP_RATIO = 0.5
    WINDOW_STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

    # Audio Features
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Skeleton Topology (Kinect v2 - 20 Joints)
    NUM_JOINTS = 20

    # Bone Connections: (Parent Index, Child Index)
    # Used to calculate explicit spatial structure vectors
    BONE_PAIRS = [
        (0, 1),
        (1, 2),
        (2, 3),  # Torso: HipCenter->Spine->ShoulderCenter->Head
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),  # Left Arm: Shoulder->Elbow->Wrist->Hand
        (2, 8),
        (8, 9),
        (9, 10),
        (10, 11),  # Right Arm: Shoulder->Elbow->Wrist->Hand
        (0, 12),
        (12, 13),
        (13, 14),
        (14, 15),  # Left Leg: Hip->Knee->Ankle->Foot
        (0, 16),
        (16, 17),
        (17, 18),
        (18, 19),  # Right Leg: Hip->Knee->Ankle->Foot
    ]

    # ==========================================
    # Model Architecture
    # ==========================================
    # Classes: 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21

    # Input Dimension Calculation:
    # 1. Relative Positions (20 joints * 3 coords) = 60
    # 2. Bone Vectors (19 bones * 3 coords) = 57
    # 3. Velocity (20 joints * 3 coords) = 60
    # 4. Acceleration (20 joints * 3 coords) = 60
    # 5. Audio MFCC = 13
    # Total = 250
    INPUT_DIM = (
        (NUM_JOINTS * 3)
        + (len(BONE_PAIRS) * 3)
        + (NUM_JOINTS * 3)
        + (NUM_JOINTS * 3)
        + N_MFCC
    )

    HIDDEN_SIZE = 128
    NUM_LAYERS = 2  # Bi-GRU Layers
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # Class Weights: 0.2 for Background (Index 0), 1.0 for Gestures (Indices 1-20)
    CLASS_WEIGHTS = [0.2] + [1.0] * 20

    # Loss Configuration
    SMOOTHING_LAMBDA = 0.15  # Weight for log-space temporal smoothing loss

    # ==========================================
    # Debugging & Development
    # ==========================================
    DEBUG = False
    DEBUG_SIZE = 20  # Number of samples to use when DEBUG is True

    @staticmethod
    def setup_directories():
        """Creates necessary directories for caching and submission."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def seed_everything(seed=None):
        """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
