import os
import torch


class Config:
    """
    Centralized configuration for the Parallel DCN-ResNet pipeline.
    Handles file paths, model architecture parameters, and training settings.
    """

    # --------------------------------------------------------------------------
    # Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs for data loading

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Data (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission (for format reference)
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "parallel_dcn_resnet.pth")

    # Cache Paths (for deterministic data processing)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # Feature Engineering
    # Prefixes for columns that should remain as raw binary (0/1) features
    BINARY_PREFIXES = ["Soil_Type", "Wilderness_Area"]

    # Debugging: Set to an integer (e.g., 10000) to limit dataset size for quick testing.
    # Set to None for full training.
    MAX_DEBUG_SAMPLES = None

    # --------------------------------------------------------------------------
    # Model Architecture: Parallel DCN-ResNet
    # --------------------------------------------------------------------------
    # ResNet Branch Settings
    RESNET_BLOCKS = 3  # Number of residual blocks
    RESNET_WIDTH = 512  # Width of the linear layers in ResNet
    RESNET_DROPOUT = 0.2  # Dropout rate within residual blocks

    # DCN Branch Settings
    DCN_LAYERS = 3  # Number of cross layers

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Optimization Budget
    BATCH_SIZE = 4096  # Large batch size for A100
    EPOCHS = 60  # Extended epochs to correct for large batch size

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
