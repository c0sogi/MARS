import os


class Config:
    """
    Configuration class for the Deep Parallel Vector-DCN-ResNet (Pre-Activation) pipeline.
    Contains all hyperparameters, file paths, and feature definitions.
    """

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Metadata parquets contain the specific train/val/test splits
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache path for processed data
    PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_data_cache.parquet")

    # --------------------------------------------------------------------------
    # Data Definitions
    # --------------------------------------------------------------------------
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Raw Continuous Features (10)
    RAW_CONTINUOUS_FEATURES = [
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

    # Binary Features (44)
    # Wilderness_Area1 to Wilderness_Area4
    WILDERNESS_FEATURES = [f"Wilderness_Area{i}" for i in range(1, 5)]
    # Soil_Type1 to Soil_Type40
    SOIL_FEATURES = [f"Soil_Type{i}" for i in range(1, 41)]

    BINARY_FEATURES = WILDERNESS_FEATURES + SOIL_FEATURES

    # Engineered Features (5)
    # These will be generated during the preprocessing pipeline
    ENGINEERED_FEATURES = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]

    # Final Continuous Features List
    # We retain raw Aspect alongside Sin/Cos as per strategy
    CONTINUOUS_FEATURES = RAW_CONTINUOUS_FEATURES + ENGINEERED_FEATURES

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    HIDDEN_DIM = 512
    NUM_BLOCKS = 4
    DROPOUT = 0.2

    # Target Classes
    # Cover_Type ranges from 1 to 7. We will map these to indices 0-6.
    NUM_CLASSES = 7

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 10
