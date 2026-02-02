import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_38"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure critical directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing
    # ==========================================
    # Sliding Window Strategy
    WINDOW_SIZE = 64
    STRIDE_TRAIN = 32
    STRIDE_TEST = 32  # 50% overlap for inference

    # Feature Engineering
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z
    # Features: Position (raw) + Velocity + Acceleration
    SKELETON_FEATURE_DIM = NUM_JOINTS * COORDS_PER_JOINT * 3  # 20 * 3 * 3 = 180
    AUDIO_MFCC_DIM = 13
    INPUT_DIM = SKELETON_FEATURE_DIM + AUDIO_MFCC_DIM  # 193 total input features

    # Class Definitions
    # 20 gestures (IDs 1-20) + 1 background class (ID 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Architecture (SHC-GKN)
    # ==========================================
    # Stage 1: Stabilized Gated Encoder
    HIDDEN_SIZE = 256  # Bi-GRU: 128 units per direction * 2
    DROPOUT_ENCODER = 0.5

    # Stage 2 & 3: TCN Refinement
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    # Monotonically increasing dilation schedule
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    DROPOUT_TCN = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50

    # Loss Configuration
    # Weighted Cross Entropy
    BACKGROUND_WEIGHT = 0.2
    # Log-Space Smoothing (Truncated MSE)
    SMOOTHING_LOSS_WEIGHT = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # Post-Processing
    # ==========================================
    # Minimum duration in frames to consider a valid gesture
    MIN_GESTURE_DURATION = 5

    # ==========================================
    # Debugging & Development
    # ==========================================
    # Set to True to train on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    @classmethod
    def get_device(cls):
        """Returns the appropriate device (GPU if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_class_weights_tensor(cls):
        """Returns the weight tensor for CrossEntropyLoss."""
        weights = [1.0] * cls.NUM_CLASSES
        weights[cls.BACKGROUND_CLASS_ID] = cls.BACKGROUND_WEIGHT
        return torch.tensor(weights, dtype=torch.float32)
