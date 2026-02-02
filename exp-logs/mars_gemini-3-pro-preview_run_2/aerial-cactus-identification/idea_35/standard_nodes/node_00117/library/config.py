import os
import torch


class Config:
    """
    Configuration for the Custom Ultra-Wide SE-RepNeXt Experiment (Idea 35).
    Acts as the central source of truth for hyperparameters, paths, and settings.
    """

    # -------------------------------------------------------------------------
    # Experiment Identity & Compute
    # -------------------------------------------------------------------------
    EXP_ID = "idea_36"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optimized for 12 vCPUs
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directory Structure
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints and caching
    WORKING_DIR = os.path.join("./working", EXP_ID)

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata CSVs (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Directory (for deterministic data processing if needed)
    CACHE_DIR = WORKING_DIR

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Preprocessing:
    # Images are loaded in [0, 1] range.
    # No resizing is applied (native 32x32).

    # Augmentation:
    # "Light Augmentation" - Only RandomHorizontalFlip and RandomVerticalFlip.
    # No rotation or color jitter.
    USE_LIGHT_AUGMENTATION = True

    # -------------------------------------------------------------------------
    # Model Architecture: Wide SE-RepVGG (Dense RepNeXt)
    # -------------------------------------------------------------------------
    # Channel dimensions for the 3 stages
    # Optimized based on Lesson 00064 (Wide SE-ResNeXt) and Lesson 00019 (Right-sizing)
    MODEL_CHANNELS = [64, 128, 256]

    # Group Cardinality for RepNeXt blocks
    # Set to 1 to create a Dense RepVGG structure (Cite Lesson 00103, Lesson 00049)
    GROUPS = 1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # Training Loop
    EPOCHS = 25
    BATCH_SIZE = 128

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler: Cosine Annealing
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 5

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    # Test Time Augmentation (TTA)
    # Strategy: Average predictions of [Original, HFlip, VFlip]
    USE_TTA = True
