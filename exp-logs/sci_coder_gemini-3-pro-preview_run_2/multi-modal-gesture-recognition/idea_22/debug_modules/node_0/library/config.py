import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Global Settings & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Directory Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 22 (DSG-CRCN)
    WORKING_DIR = "./working/idea_22"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    # Gesture Vocabulary (1-20). 0 is reserved for Background.
    NUM_CLASSES = 21

    # Feature Selection: 12 Upper-Body Joints
    # Indices based on Kinect Skeleton format provided in description
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Input Dimensions
    # Skeleton: 12 joints * 3 coords (X,Y,Z) = 36
    # Velocity: 12 joints * 3 coords = 36 (derived)
    # Audio: 13 MFCCs
    INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + 13

    # Normalization
    SCALE_FACTOR = 0.001  # Convert mm to meters
    CENTER_JOINT_IDX = 0  # HipCenter

    # Sequence Handling
    MAX_LEN = 3000  # Safe upper bound for padding

    # =========================================================================
    # Model Architecture: DSG-CRCN
    # =========================================================================
    # Stage 1: Bi-LSTM Encoder
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2

    # Stage 2 & 3: Dual-Scale Gated Refinement
    # Channels for the refinement stages (concatenated from branches)
    REFINEMENT_CHANNELS = 256
    REFINEMENT_LAYERS = 10

    # Dual-Scale Branch Settings
    KERNEL_SIZE_GLOBAL = 3  # Dilated branch
    KERNEL_SIZE_LOCAL = 3  # Fixed branch (dilation=1)
    DROPOUT = 0.3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # Loss Weights
    # Strict ratio: 0.1 (Background) : 1.0 (Gesture)
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Multi-Task Loss Components
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_BND = 1.0  # Boundary regression
    LOSS_WEIGHT_TMSE = 0.15  # Unclamped Probability-Space Smoothing

    # =========================================================================
    # Utility Functions
    # =========================================================================
    @classmethod
    def setup(cls):
        """Creates necessary working directories and sets random seeds."""
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Deterministic Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for cudnn
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_class_weights_tensor(cls):
        """Returns the class weights as a tensor on the configured device."""
        return torch.tensor(cls.CLASS_WEIGHTS, dtype=torch.float32).to(cls.DEVICE)

    # Gesture Mapping for reference and decoding
    GESTURE_MAP = {
        "vattene": 1,
        "vieniqui": 2,
        "perfetto": 3,
        "furbo": 4,
        "cheduepalle": 5,
        "chevuoi": 6,
        "daccordo": 7,
        "seipazzo": 8,
        "combinato": 9,
        "freganiente": 10,
        "ok": 11,
        "cosatifarei": 12,
        "basta": 13,
        "prendere": 14,
        "noncenepiu": 15,
        "fame": 16,
        "tantotempo": 17,
        "buonissimo": 18,
        "messidaccordo": 19,
        "sonostufo": 20,
    }

    # Reverse mapping for debugging
    ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}
