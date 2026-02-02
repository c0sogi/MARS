import os
import torch


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and saving models
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    # High-Density Patching settings
    PATCH_SIZE = 50
    STRIDE = 10  # Small stride for high overlap/density

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Model Architecture Hyperparameters (S-RDN)
    # =========================================================================
    IN_CHANNELS = 1
    OUT_CHANNELS = 1

    # RDN specific parameters
    GROWTH_RATE = 32  # Number of feature maps added per layer in a dense block
    NUM_RDN_BLOCKS = 8  # Number of Residual Dense Blocks (RDB)
    NUM_LAYERS_PER_BLOCK = 4  # Number of convolutional layers inside each RDB
    NUM_FEATURES = 64  # Number of features in the global residual path

    # Stability Mechanism
    RESIDUAL_SCALE = (
        0.1  # Scaling factor for residual branches to prevent signal explosion
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4

    # Optimization and Stability
    GRADIENT_CLIP_VALUE = 1.0  # Max norm for gradient clipping
    WEIGHT_DECAY = 1e-8

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-5

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
