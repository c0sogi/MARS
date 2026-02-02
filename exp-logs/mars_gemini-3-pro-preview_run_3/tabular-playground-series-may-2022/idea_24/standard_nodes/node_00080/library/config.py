import os


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    SUBMISSION_DIR = "./submission"

    # Input Files (using metadata for stratified splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet format preferred over pickle)
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    CACHE_META_PATH = os.path.join(WORKING_DIR, "metadata.npy")  # For vocab sizes, etc.

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Configuration
    # ==========================================
    # Feature Engineering
    EMBED_DIM = 16

    # Debugging / Development
    # Set DEBUG = True to use a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000

    # ==========================================
    # Model Architecture: Safe-Spectrum Parallel Funnel Ensemble
    # ==========================================
    # 5 Independent Streams
    # Structure: List of dicts, each defining the hidden layers and dropout for a stream
    # Stream 1 & 2: Anchors (Standard Funnel, Dropout 0.20)
    # Stream 3: Capacity Variant (Wide Funnel, Dropout 0.25)
    # Stream 4: Safe-Aggressive Variant (Standard Funnel, Dropout 0.15)
    # Stream 5: Conservative Variant (Standard Funnel, Dropout 0.30)

    MODEL_STREAMS = [
        {"layers": [512, 256, 128], "dropout": 0.20},  # Stream 1
        {"layers": [512, 256, 128], "dropout": 0.20},  # Stream 2
        {"layers": [1024, 512, 256], "dropout": 0.25},  # Stream 3
        {"layers": [512, 256, 128], "dropout": 0.15},  # Stream 4
        {"layers": [512, 256, 128], "dropout": 0.30},  # Stream 5
    ]

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 30  # Sufficient for super-convergence

    # Optimizer (AdamW)
    WEIGHT_DECAY = 1e-4

    # Scheduler (OneCycleLR)
    MAX_LR = 1e-2
    PCT_START = 0.3  # Default for OneCycle, can be tuned
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 5

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
