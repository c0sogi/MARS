import os
import torch


class Config:
    """
    Configuration class for the Spatial Symmetry-Difference Siamese Network.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Fallback for preprocessed images from previous successful runs
    PREPROCESSED_DIR = "./working/idea_2/processed_images"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    # Input Strategy: 3 Channels [Image, Age, Implant]
    IMG_SIZE = 512
    IN_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size set conservatively for 768x768 Siamese network on A100
    BATCH_SIZE = 8
    NUM_EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss settings
    # Aggressive positive weighting to handle 1:47 imbalance
    POS_WEIGHT = 47.0

    # Scheduler
    T_MAX = 10  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12
    PIN_MEMORY = False
