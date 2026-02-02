import os
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"
    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_6"

    # Input Data (Parquet files from metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet_model.pth")

    # Caching Paths (for processed numpy arrays)
    # Updated filenames to force re-processing with new features
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X_v2.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y_v2.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X_v2.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y_v2.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X_v2.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids_v2.npy")

    # ==========================================
    # Column Definitions
    # ==========================================
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Raw Continuous Features (Full names)
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

    # Binary Features (One-Hot Encoded in source)
    # Wilderness_Area1 to Wilderness_Area4
    # Soil_Type1 to Soil_Type40
    BINARY_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
        f"Soil_Type{i}" for i in range(1, 41)
    ]

    # Engineered Geometric Features
    NEW_FEATURES = ["Euclidean_Distance_To_Hydrology", "Absolute_Hydrology_Elevation"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Target classes are 1-7. We map to 0-6 internally.
    NUM_CLASSES = 7

    # Architecture
    HIDDEN_DIM = 256
    NUM_RES_BLOCKS = 3
    NUM_CROSS_LAYERS = 3
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 2048  # A100 allows large batches
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    PATIENCE = 5  # Early stopping patience
    FACTOR = 0.5  # ReduceLROnPlateau factor
    MIN_LR = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
