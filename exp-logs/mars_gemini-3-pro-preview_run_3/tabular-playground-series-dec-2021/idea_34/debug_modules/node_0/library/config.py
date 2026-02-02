import os
import torch


class Config:
    """
    Central configuration for the Tri-Branch Wide-Deep-Cross Network experiment.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory for this specific idea (Idea 34)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_34")

    # Input Files (Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # Caching Paths for Processed Data
    TRAIN_PROCESSED_PATH = os.path.join(CACHE_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(CACHE_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(CACHE_DIR, "test_processed.parquet")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Disable strict determinism for performance as per strategy
    DETERMINISTIC_CUDNN = False

    # ==========================================
    # Data Engineering Constants
    # ==========================================
    # Column names used for feature engineering
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    COL_ASPECT = "Aspect"
    COL_ELEVATION = "Elevation"
    COL_VERT_DIST_HYDRO = "Vertical_Distance_To_Hydrology"
    COL_HORZ_DIST_HYDRO = "Horizontal_Distance_To_Hydrology"
    COL_HORZ_DIST_ROAD = "Horizontal_Distance_To_Roadways"
    COL_HORZ_DIST_FIRE = "Horizontal_Distance_To_Fire_Points"

    # ==========================================
    # Model Architecture
    # ==========================================
    # Tri-Branch Wide-Deep-Cross Network settings
    HIDDEN_DIM = 512
    NUM_RESNET_BLOCKS = 4  # Deep Backbone depth
    NUM_CROSS_LAYERS = 3  # Asymmetric Cross Branch depth
    DROPOUT_RATE = 0.2  # Regularization

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW default
    LABEL_SMOOTHING = 0.1  # Regularization

    # Scheduler
    SCHEDULER_FACTOR = 0.1  # Aggressive decay
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # System
    NUM_WORKERS = 4  # Based on 12 vCPUs available

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
