import os
import torch


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    # Root directory for input data (read-only)
    INPUT_DIR = "./input"

    # Directory for generated metadata (read-only)
    METADATA_DIR = "./metadata"

    # Working directory for artifacts, cache, and models
    # Using a specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_15"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache directory for processed tensors/dataframes
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Path to save the best model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Path to save the final submission file
    SUBMISSION_SAVE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Preprocessing Parameters
    # ==========================================
    # Features derived from:
    # 1. Cn0DbHz: mean, std, min, max (4)
    # 2. SvElevationDegrees: mean, std, min, max (4)
    # 3. SvAzimuthDegrees: mean_sin, mean_cos (2)
    # 4. Signal Moments: weighted_sin, weighted_cos (2)
    # 5. SatCount (1)
    # 6. RawPseudorangeUncertaintyMeters: mean (1)
    NUM_FEATURES = 14

    # Target variables: Delta East, Delta North (Meters)
    NUM_CLASSES = 2

    # Downsampling factors for Multi-Scale Deep Supervision
    # Corresponding to the output of each decoder block before upsampling
    # Depth 0 (Final), Depth 1, Depth 2, Depth 3
    DEEP_SUPERVISION_SCALES = [1, 2, 4, 8]

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    # Base number of filters in the first encoder block
    BASE_FILTERS = 64

    # Depth of the U-Net (number of encoder/decoder blocks)
    MODEL_DEPTH = 4

    # Kernel size for 1D convolutions
    KERNEL_SIZE = 3

    # Dropout probability
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 16  # Adjusted for sequence data memory usage
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss weights for deep supervision heads
    # Weights correspond to scales [1, 2, 4, 8]
    # Higher resolution gets higher weight
    LOSS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

    # ==========================================
    # Setup
    # ==========================================
    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately when module is imported
Config.setup()
