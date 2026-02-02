import os
import torch


class Config:
    """
    Centralized configuration for the Asymmetric Parallel Low-Rank-DCN-ResNet strategy.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_41"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Data (Metadata Parquet Files)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pth")

    # Cache Paths for Processed Data (using .npy for efficiency)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Observed classes in Train/Val: 1, 2, 3, 4, 6, 7 (Class 5 is missing)
    # We map these to 0-5 for internal processing
    NUM_CLASSES = 6

    CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}

    INVERSE_CLASS_MAPPING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7}

    # Feature Engineering Flags
    USE_QUANTILE_TRANSFORM = True
    QUANTILE_OUTPUT_DIST = "normal"

    # --------------------------------------------------------------------------
    # Model Architecture: Asymmetric Parallel Low-Rank-DCN-ResNet
    # --------------------------------------------------------------------------
    # Branch 1: Low-Rank Factorized DCN
    DCN_RANK = 4  # Rank-4 decomposition (Lesson 00032/00053)
    DCN_LAYERS = 3  # Asymmetric depth (Lesson 00071)

    # Branch 2: ResNet Backbone
    RESNET_BLOCKS = 4  # Fixed at 4 blocks for stability (Lesson 00073/00078)
    HIDDEN_DIM = 512  # Capacity (Lesson 00029)

    # General
    DROPOUT_RATE = 0.2  # Regularization (Lesson 00056)

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60  # Fixed budget (Lesson 00022)

    # Optimizer: AdamW (Lesson 00058)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler: ReduceLROnPlateau (Aggressive)
    SCHEDULER_FACTOR = 0.1  # Aggressive decay (Lesson 00068)
    SCHEDULER_PATIENCE = 3

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # --------------------------------------------------------------------------
    # Hardware / System
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Disable strict determinism for performance (Lesson 00070)
    CUDNN_DETERMINISTIC = False
