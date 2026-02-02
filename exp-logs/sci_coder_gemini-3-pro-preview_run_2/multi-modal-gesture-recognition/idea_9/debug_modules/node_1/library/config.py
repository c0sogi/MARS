import os
import random
import numpy as np
import torch


class Config:
    """
    Centralized configuration for the Masked Dual-Stage Cascaded Recurrent-Convolutional Network (MD-CRCN).
    Handles paths, hyperparameters, domain constants, and reproducibility settings.
    """

    # =========================================================================
    # 1. File System & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Artifact Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Create directories if they don't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # =========================================================================
    # 2. Reproducibility
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed seeds for reproducibility across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # =========================================================================
    # 3. Domain Constants & Data Processing
    # =========================================================================
    # Gesture Vocabulary
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
    # Reverse mapping for decoding predictions
    IDX_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

    # Class Definitions
    # 20 Gestures + 1 Background class (Index 0)
    NUM_CLASSES = 21
    BACKGROUND_LABEL = 0

    # Feature Selection: Upper Body Only
    # Indices correspond to standard Kinect skeleton format
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_SKELETON_JOINTS = len(UPPER_BODY_JOINTS)

    # Feature Dimensions
    # Per Joint: 3 Position (x,y,z) + 3 Velocity (dx,dy,dz)
    JOINT_FEATS = 6
    # Audio: Mel-frequency cepstral coefficients
    AUDIO_MFCC_DIM = 13

    # Total Input Feature Dimension
    # (12 joints * 6 features) + 13 audio features = 85
    INPUT_DIM = (NUM_SKELETON_JOINTS * JOINT_FEATS) + AUDIO_MFCC_DIM

    # Audio Processing
    AUDIO_SAMPLE_RATE = 16000

    # =========================================================================
    # 4. Model Architecture (MD-CRCN)
    # =========================================================================
    # Stage 1: Recurrent Encoder (Bi-LSTM)
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # Stage 2 & 3: Temporal Convolutional Network (TCN)
    # 10 layers of dilated convolutions
    TCN_NUM_CHANNELS = [256] * 10
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3

    # =========================================================================
    # 5. Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 15
    WEIGHT_DECAY = 1e-4

    # Loss Function Configuration
    # Aggressive weighting: 0.1 for Background, 1.0 for all Gestures
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # T-MSE Loss for probability smoothing
    TMSE_THRESHOLD = 4.0

    # =========================================================================
    # 6. Inference & Post-Processing
    # =========================================================================
    # Median filter kernel size for smoothing label predictions
    MEDIAN_FILTER_KERNEL_SIZE = 7
