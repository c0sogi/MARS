import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for this specific idea (Idea 23)
    PROJECT_NAME = "idea_23"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Cache Directory for deterministic data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 256
    NUM_WORKERS = 4

    # Input Channels: 3 (Ash) + 3 (Temporal Diff) = 6
    IN_CHANNELS = 6

    # Ash Composite Normalization Bounds (Kelvin)
    # Used to normalize Ash channels to [0, 1]
    # Band 15 - Band 14
    ASH_RED_MIN = -4.0
    ASH_RED_MAX = 2.0
    # Band 14 - Band 11
    ASH_GREEN_MIN = -4.0
    ASH_GREEN_MAX = 5.0
    # Band 14
    ASH_BLUE_MIN = 243.0
    ASH_BLUE_MAX = 303.0

    # Raw Band Difference Normalization Bounds (Kelvin)
    # Used to normalize temporal differences (t4 - t3) for Bands 11, 14, 15 to [0, 1]
    DIFF_MIN = -5.0
    DIFF_MAX = 5.0

    # ==========================================
    # Model Architecture
    # ==========================================
    ENCODER_NAME = "convnext_tiny"
    ENCODER_PRETRAINED = True

    # Metadata features to inject: row_min (lat), col_min (lon), timestamp (time)
    METADATA_FEATURE_DIM = 3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Loss Weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # ==========================================
    # Reproducibility & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Toggle to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500

    @classmethod
    def display(cls):
        """Prints the configuration."""
        print("=" * 30)
        print(f"Configuration: {cls.PROJECT_NAME}")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
