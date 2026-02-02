import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # 1. General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging
    DEBUG_SAMPLE_SIZE = 10000  # Number of rows to use if DEBUG is True

    # -------------------------------------------------------------------------
    # 2. Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata Parquet)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "parallel_low_rank_dcn_resnet.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 3. Data & Feature Engineering
    # -------------------------------------------------------------------------
    # Raw continuous features to be standardized
    CONTINUOUS_FEATURES = [
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

    # Target Configuration
    # Classes present in data: 1, 2, 3, 4, 6, 7 (Class 5 is missing)
    # We map them to contiguous integers 0-5 for training
    NUM_CLASSES = 6
    LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    # -------------------------------------------------------------------------
    # 4. Model Architecture (Parallel Low-Rank DCN-ResNet)
    # -------------------------------------------------------------------------
    # DCN Branch
    DCN_RANK = 16  # Rank r for Low-Rank Factorized Cross Layer

    # ResNet Branch
    RESNET_HIDDEN_DIM = 512
    RESNET_NUM_BLOCKS = 2  # Depth of the backbone

    # General
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # 5. Optimization & Training
    # -------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3

    # Scheduler: ReduceLROnPlateau
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MODE = "max"  # We monitor validation accuracy

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # -------------------------------------------------------------------------
    # 6. Hardware & Execution
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
