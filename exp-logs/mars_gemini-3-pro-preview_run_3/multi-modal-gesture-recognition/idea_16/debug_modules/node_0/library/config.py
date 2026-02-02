import os
import torch


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 16
    WORK_DIR = "./working/idea_16"

    # Cache directory for processed features (parquet/npz)
    CACHE_DIR = os.path.join(WORK_DIR, "cache")

    # Model checkpoints and submission
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    # Windowing
    WINDOW_SIZE = 64
    TRAIN_STRIDE = 8  # Overlap for training data generation (augmentation)
    TEST_STRIDE = 32  # 50% overlap for inference

    # Skeleton
    NUM_JOINTS = 20
    JOINT_DIMS = 3  # x, y, z
    USE_VELOCITY = True
    USE_ACCELERATION = True

    # Audio
    AUDIO_N_MFCC = 13

    # Input Dimension Calculation
    # Skeleton: 20 joints * 3 coords * (1 pos + 1 vel + 1 acc) = 180
    SKELETON_DIM = (
        NUM_JOINTS * JOINT_DIMS * (1 + int(USE_VELOCITY) + int(USE_ACCELERATION))
    )
    AUDIO_DIM = AUDIO_N_MFCC
    INPUT_DIM = SKELETON_DIM + AUDIO_DIM  # 180 + 13 = 193

    # Classes: 20 gestures + 1 background (index 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Augmentation
    AUG_ROTATION_RANGE = 15  # Degrees
    AUG_SCALE_RANGE = 0.1  # +/- 10%

    # ==========================================
    # Model Architecture (RD-KRN)
    # ==========================================
    # Stage 1: Bi-GRU
    GRU_HIDDEN_DIM = 128
    GRU_NUM_LAYERS = 2

    # Stage 2 & 3: Deep-Field Refinement (TCN)
    # Dilation schedule: 1, 2, 4, 8, 16, 32 (Receptive field maximization)
    TCN_NUM_LAYERS = 6
    TCN_KERNEL_SIZE = 3
    TCN_FEATURE_DIM = 128
    TCN_DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 12

    # Loss Weights
    # Background class gets 0.2 weight, others 1.0
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

    # Smoothing Loss
    MSE_LOSS_WEIGHT = 0.15

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup_directories(cls):
        """Ensures that necessary working directories exist."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
