import os
import torch


class Config:
    """
    Global configuration for paths, device, and environment settings.
    """

    # Reproducibility
    SEED = 42

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths (using pre-generated metadata parquets)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Paths
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache Directory for deterministic processing
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)


class DataConfig:
    """
    Configuration for feature engineering and dataset structure.
    """

    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # 1. Raw Continuous Features (10 columns)
    RAW_CONT_COLS = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrol",
        "Vertical_Distance_To_Hydrolog",
        "Horizontal_Distance_To_Roadwa",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_P",
    ]

    # 2. Binary Features (44 columns)
    # Wilderness_Area1 to Wilderness_Area4
    WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
    # Soil_Type1 to Soil_Type40
    SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]

    BINARY_COLS = WILDERNESS_COLS + SOIL_COLS

    # 3. New Engineered Features (5 columns)
    # Defined in "Augmented Physics-Informed Engineering"
    NEW_CONT_COLS = [
        "Aspect_Sin",  # Cyclical Augmentation
        "Aspect_Cos",  # Cyclical Augmentation
        "Hydrology_Distance",  # Geometric Magnitude (Euclidean)
        "Hydrology_Elevation",  # Directional Preservation (Elevation - Vertical_Dist)
        "Mean_Amenities_Dist",  # Global Context
    ]

    # Combined Feature Lists
    CONT_COLS = RAW_CONT_COLS + NEW_CONT_COLS

    # Input Dimension Calculation
    # 10 (Raw) + 5 (New) = 15 Continuous
    # 44 Binary
    # Total Input Dimension = 59
    INPUT_DIM = len(CONT_COLS) + len(BINARY_COLS)

    # Target Classes
    # Cover_Type ranges from 1 to 7.
    NUM_CLASSES = 7


class ModelConfig:
    """
    Hyperparameters for the Parallel Low-Rank DCN-ResNet.
    """

    INPUT_DIM = DataConfig.INPUT_DIM
    HIDDEN_DIM = 512  # Wide ResNet Backbone
    DCN_RANK = 16  # Low-Rank Factorization (r << d)
    DROPOUT = 0.0  # No dropout as per "Wide ResNet" spec
    ACTIVATION = "relu"  # Standard ReLU
    NUM_CLASSES = DataConfig.NUM_CLASSES


class TrainConfig:
    """
    Training loop configuration and optimization hyperparameters.
    """

    # Budget
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimization
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    # Cosine Annealing over the full 60 epochs
    SCHEDULER_TYPE = "cosine"

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Validation
    VAL_CHECK_INTERVAL = 1  # Validate every epoch
