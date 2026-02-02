import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration for the Root-Centric Moderate-Capacity Network (RC-MCN).
    Centralizes all hyperparameters, file paths, and architectural constants.
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

    # Working directory for this specific idea (Idea 44)
    WORKING_DIR = "./working/idea_44"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Augmentation
    # ==========================================
    # Windowing strategy
    WINDOW_SIZE = 64
    STRIDE = 32  # Moderate stride to avoid overfitting

    # Feature Dimensions
    # Skeleton: 20 joints * 3 coords (X,Y,Z) * 3 derivatives (Pos, Vel, Acc)
    NUM_JOINTS = 20
    SKELETON_INPUT_DIM = NUM_JOINTS * 3 * 3  # 180

    # Audio: MFCC features
    AUDIO_MFCC_DIM = 13

    # Total Input Dimension for Early Fusion
    INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_MFCC_DIM  # 193

    # Classes: 20 Gestures + 1 Background (Class 0)
    NUM_CLASSES = 21

    # ==========================================
    # Model Architecture (RC-MCN)
    # ==========================================
    # Stage 1: Encoder
    HIDDEN_DIM = 192  # 96 per direction for Bi-GRU
    DROPOUT_ENCODER = 0.4

    # Stage 2 & 3: TCN Refinement
    TCN_NUM_CHANNELS = [NUM_CLASSES] * 5  # Keep channel dim constant
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2  # DROPOUT_TCN
    # Monotonically increasing dilations: Receptive field = 1 + 2*(1+2+4+8+16) = 63
    TCN_DILATIONS = [1, 2, 4, 8, 16]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function Weights
    BACKGROUND_CLASS_WEIGHT = 0.2

    # Deep Supervision & Smoothing
    # L_total = L_p1 + L_p2 + L_p3 + Smoothing
    SMOOTHING_WEIGHT = 0.15
    SMOOTHING_THRESHOLD = 1.0  # Truncated MSE threshold

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    MIN_DURATION = 5  # Minimum frames for a valid gesture

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def setup_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
