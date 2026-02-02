import os
import torch


class Config:
    """
    Configuration class for the Cactus Identification task.
    Implements the settings for the Custom Ultra-Wide SE-RepNeXt architecture.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this experiment idea
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # DataLoader settings
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Model Architecture: Custom Ultra-Wide SE-RepNeXt
    # =========================================================================
    # Channel widths for the 3 stages (Ultra-Wide configuration)
    STAGE_CHANNELS = [96, 192, 384]

    # Group cardinality for the RepNeXt blocks
    GROUPS = 32

    # Squeeze-and-Excitation settings
    USE_SE = True
    SE_REDUCTION = 16  # Reduction ratio for SE blocks

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # Optimization
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Scheduler (Cosine Annealing)
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # =========================================================================
    # Hardware & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Base seed for operations not covered by the ensemble loop
    BASE_SEED = 42
