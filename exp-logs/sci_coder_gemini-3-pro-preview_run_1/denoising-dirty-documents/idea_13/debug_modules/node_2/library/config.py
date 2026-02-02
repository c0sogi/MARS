import os
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache paths for deterministic data processing
    # Using .npy format for efficient numpy array storage
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npz")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    IMG_SIZE = 320  # Crop size for training
    FULL_IMG_SIZE = (420, 540)  # Approximate full size for reference/inference padding
    CHANNELS = 1  # Grayscale

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16
    EPOCHS = 1000
    LR = 1e-3

    # Ensemble Strategy: 10 Independent Models
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    # Optimizer and Scheduler settings
    WEIGHT_DECAY = 0.0  # Standard for Adam in this context
    T_MAX = EPOCHS  # For Cosine Annealing

    # =========================================================================
    # System Settings
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Model Architecture Settings
    # =========================================================================
    # U-Net depth and filters
    ENCODER_FILTERS = [32, 64, 128, 256, 512]

    @classmethod
    def get_model_path(cls, seed):
        """Returns the path to save/load the model for a specific seed."""
        return os.path.join(cls.WORKING_DIR, f"model_seed_{seed}.pth")
