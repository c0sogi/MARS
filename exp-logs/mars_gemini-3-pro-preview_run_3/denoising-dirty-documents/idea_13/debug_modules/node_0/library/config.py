import os
import torch


class Config:
    """
    Configuration class for the Ensemble of Zero-Initialized Deep Residual Networks (EZ-ResDnCNN).
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Metadata CSV Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    PATCH_SIZE = 50
    STRIDE = 10  # Low stride for high-density patch extraction
    AUGMENTATION = True  # Enable geometric augmentations (flips/rotations)
    NORMALIZE = True  # Normalize pixel intensities to [0, 1]

    # Caching Paths (using .npy for fast I/O)
    # These paths are used to store the dense patch datasets after extraction
    CACHE_TRAIN_PATCHES = os.path.join(WORKING_DIR, "train_patches.npy")
    CACHE_TRAIN_TARGETS = os.path.join(WORKING_DIR, "train_targets.npy")
    CACHE_VAL_PATCHES = os.path.join(WORKING_DIR, "val_patches.npy")
    CACHE_VAL_TARGETS = os.path.join(WORKING_DIR, "val_targets.npy")

    # ==========================================
    # Model Architecture (EZ-ResDnCNN)
    # ==========================================
    IN_CHANNELS = 1
    OUT_CHANNELS = 1  # Network predicts the noise residual
    NUM_FEATURES = 64
    NUM_RES_BLOCKS = 20  # Deep linear stack of residual blocks
    KERNEL_SIZE = 3
    PADDING = 1
    # Zero-Gamma Initialization: Initialize the last BN gamma in each residual block to 0
    # This ensures the block acts as identity at initialization, stabilizing deep training.
    ZERO_GAMMA_INIT = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    ENSEMBLE_SIZE = 5  # Number of independent models to train

    # Optimization
    BATCH_SIZE = 128
    NUM_EPOCHS = 60  # Sufficient epochs given the dense data
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 1e-6

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware & Runtime
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Utilizing available vCPUs for data loading

    # ==========================================
    # Inference / TTA
    # ==========================================
    # Test Time Augmentation: 8 geometric transformations (D4 Dihedral Group)
    # (Identity, Rotate90, Rotate180, Rotate270, FlipLR, FlipUD, FlipDiag, FlipAntiDiag)
    TTA_STEPS = 8
