import os
import torch


class Config:
    """
    Configuration class for the Projected Dual-Polarity Hybrid-SE CNN (PDPH-SE-CNN).
    Centralizes hyperparameters, file paths, and model specifications.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_41"  # Cache directory for processed data
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IN_CHANNELS = 3  # HH, HV, Average((HH+HV)/2)

    # Augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5

    # -------------------------------------------------------------------------
    # Model Architecture (PDPH-SE-CNN)
    # -------------------------------------------------------------------------
    # Backbone: Plain CNN with 4 stages
    # Width Strategy: Expand early, cap width to prevent overfitting
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Activation
    LEAKY_RELU_SLOPE = 0.1  # Preserves negative values (shadows)

    # Projected Dual-Polarity Readout
    # Compress channel dim before pooling to control parameter count
    # 128 channels -> 64 channels -> (Max Pool + Min Pool) = 128 features
    PROJECTED_DIM = 64

    # Classification Head
    DROPOUT_RATE = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    NUM_EPOCHS = 75
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-2  # L2 Regularization (standard for AdamW)

    # Early Stopping
    PATIENCE = 12

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
