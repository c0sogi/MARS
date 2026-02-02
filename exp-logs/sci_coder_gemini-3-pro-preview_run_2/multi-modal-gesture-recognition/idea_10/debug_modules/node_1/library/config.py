import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Main working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_10"

    # Subdirectories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEED = 42

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

    # Inverse map for submission
    ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

    # Classes: 0 (Background) + 20 Gestures = 21
    NUM_CLASSES = 21

    # Feature Selection: 12 Upper-Body Joints
    # Indices based on the provided dataset description order:
    # 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Input Dimensions
    # Skeleton: 12 joints * 3 coords (x,y,z) = 36 features
    # Velocity: 12 joints * 3 coords = 36 features (derived)
    # Audio: MFCCs (e.g., 13 or 20 coefficients)
    SKELETON_INPUT_SIZE = len(SELECTED_JOINTS) * 3  # 36
    VELOCITY_INPUT_SIZE = len(SELECTED_JOINTS) * 3  # 36
    AUDIO_INPUT_SIZE = 13  # Number of MFCCs

    # Total input dimension for the model (before fusion) depends on implementation
    # but here we define the components.

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Gated Multimodal Unit
    FUSION_HIDDEN_SIZE = 128

    # Stage 1: Bi-LSTM
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # Stage 2 & 3: MS-TCN
    TCN_NUM_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.3
    # 10 levels: dilation 2^0 to 2^9 (1 to 512)
    TCN_NUM_LAYERS = 10

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 8  # Sequences can be long, keep batch size moderate
    NUM_EPOCHS = 60

    # Early Stopping
    PATIENCE = 10

    # Loss Weights
    # 0.1 for Background (Class 0), 1.0 for Gestures (Classes 1-20)
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Deep Supervision Weights
    # L_total = L_gen + L_ref1 + L_ref2
    LOSS_LAMBDA_GEN = 1.0
    LOSS_LAMBDA_REF1 = 1.0
    LOSS_LAMBDA_REF2 = 1.0

    # Smoothing Loss (TMSE)
    TMSE_WEIGHT = 0.15
    TMSE_THRESHOLD = 4.0  # Variance threshold

    # =========================================================================
    # Inference / Post-processing
    # =========================================================================
    MEDIAN_FILTER_KERNEL = 7  # Size of median filter window

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets random seeds."""
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_class_weights_tensor(cls, device):
        """Returns the class weights as a tensor on the specified device."""
        return torch.tensor(cls.CLASS_WEIGHTS, dtype=torch.float32).to(device)
