import os


class Config:
    """
    Configuration for the Spatially-Contextualized Wide-Body Network (SC-WBN).
    Defines all file paths, hyperparameters, and global constants.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (artifacts, checkpoints, cache)
    WORKING_DIR = "./working/idea_30"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache File for processed tensors
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # ==========================================
    # 2. DATA SPECIFICATIONS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IMG_CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)

    # Global Statistics from EDA (for Global Min-Max Scaling)
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806
    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # ==========================================
    # 3. MODEL HYPERPARAMETERS
    # ==========================================
    # Architecture
    BACKBONE_FILTERS = 128  # Wide backbone capacity
    READOUT_CHANNELS = 64  # Spatial-Context Bottleneck output depth
    DENSE_INPUT_DIM = 1024  # 4x4x64 flattened

    # Regularization
    DROPOUT_RATE = 0.5  # High dropout for regularization

    # ==========================================
    # 4. TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # Optimization ("Low and Slow")
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4  # Conservative start
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    NUM_EPOCHS = 50  # Sufficient for convergence with early stopping
    PATIENCE = 12  # Early stopping patience

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # ==========================================
    # 5. DEBUGGING & CONTROLS
    # ==========================================
    # Set to True to run on a small subset of data for testing pipeline
    DEBUG = False

    # If DEBUG is True, use this many samples
    MAX_DEBUG_SAMPLES = 100

    # Number of workers for data loading
    NUM_WORKERS = 2
