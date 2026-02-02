import os
import torch


class Config:
    """
    Configuration class for the Zero-Initialized Deep Residual Denoising Network (ZI-ResDnCNN).
    Defines all hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    # Use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of data loading workers (12 vCPUs available, 4 is usually optimal for IO overhead)
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache file paths for processed patches (Numpy format)
    TRAIN_PATCHES_CACHE = os.path.join(WORKING_DIR, "train_patches.npy")
    TRAIN_TARGETS_CACHE = os.path.join(WORKING_DIR, "train_targets.npy")
    VAL_PATCHES_CACHE = os.path.join(WORKING_DIR, "val_patches.npy")
    VAL_TARGETS_CACHE = os.path.join(WORKING_DIR, "val_targets.npy")

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "zi_resdncnn_best_model.pth")

    # Submission paths
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Patch extraction settings
    PATCH_SIZE = 50
    STRIDE = 10  # Low stride for high overlap to increase data density

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # ZI-ResDnCNN specific parameters
    NUM_BLOCKS = 20  # Deep stack of residual blocks
    NUM_CHANNELS = 64  # Number of feature maps
    KERNEL_SIZE = 3  # 3x3 Convolutions
    PADDING = 1  # Maintain spatial resolution
    USE_ZERO_GAMMA = True  # Enable Zero-Gamma Initialization for stability

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-6  # For Cosine Annealing Scheduler
    WEIGHT_DECAY = 1e-8  # Slight regularization
    EARLY_STOPPING_PATIENCE = 10

    # ==========================================
    # Inference Hyperparameters
    # ==========================================
    # Geometric Self-Ensemble (Test-Time Augmentation)
    # If True, averages predictions from 8 geometric transformations (flips/rotations)
    USE_TTA = True

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    # Number of patches to use when DEBUG is True
    DEBUG_SUBSET_SIZE = 2000
