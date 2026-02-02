import os
import torch


class Config:
    """
    Central configuration for the Stabilized Selective-Hierarchical SE-CNN.
    """

    # --------------------------------------------------------------------------
    # Directories and Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to ensure isolation
    WORKING_DIR = "./working/idea_20"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 75
    # Input channels: HH, HV, and Average((HH+HV)/2)
    INPUT_CHANNELS = 3

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick pipeline testing
    DEBUG = False
    DEBUG_SIZE = 100

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Architecture: Custom 4-Stage Attentive CNN
    # Width Strategy: Early Expansion (64 -> 128 -> 128 -> 128)
    CHANNEL_WIDTHS = [64, 128, 128, 128]

    # Squeeze-and-Excitation settings
    USE_SE = True
    SE_REDUCTION = 16

    # Classification Head
    FC_DROPOUT = 0.5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5

    # Optimization
    BATCH_SIZE = 64
    EPOCHS = 100

    # Optimizer: Adam with constant learning rate
    LEARNING_RATE = 1e-3

    # Regularization
    # L2 Weight Decay to prevent confident errors
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    # Restricted patience to prevent validation overfitting
    PATIENCE = 10

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    # Explicitly disable TTA (Test Time Augmentation)
    USE_TTA = False
