import os


class Config:
    """
    Configuration class for the Deep Parallel Vector-DCN-ResNet pipeline.
    Defines hyperparameters, file paths, and feature groups.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths (using pre-generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Continuous Features to be standardized
    CONTINUOUS_COLS = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    # Binary Features (Already One-Hot/Binary in source)
    # Wilderness Areas (4) and Soil Types (40)
    BINARY_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
        f"Soil_Type{i}" for i in range(1, 41)
    ]

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Deep Parallel Vector-DCN-ResNet (Full Pre-Activation)
    HIDDEN_DIM = 512
    NUM_BLOCKS = 4
    DROPOUT = 0.2
    NUM_CLASSES = 7  # Classes 1-7 mapped to 0-6

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # --------------------------------------------------------------------------
    # Runtime / Debugging
    # --------------------------------------------------------------------------
    # Number of workers for DataLoader
    NUM_WORKERS = 4

    # Limit training samples for debugging (Set to None for full training)
    # This allows flexibility as requested
    MAX_TRAIN_SAMPLES = None
