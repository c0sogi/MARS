import os
import torch


class Config:
    # ==========================================
    # Global Reproducibility & Compute
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, setting workers accordingly
    NUM_WORKERS = 4

    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input Data (Using metadata splits as required)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output & Caching
    # Specific working directory for this idea
    WORKING_DIR = "./working/idea_36"
    CACHE_DIR = WORKING_DIR

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpointing
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "target"

    # Feature Engineering Constants
    EMBEDDING_DIM = 16

    # Categorical Columns:
    # f_27 is handled via character decomposition in the dataset class.
    # f_29 and f_30 are treated as categorical.
    CATEGORICAL_COLS = ["f_29", "f_30"]

    # ==========================================
    # Model Architecture: SR-PFE
    # ==========================================
    # Selective-Residual Parallel Funnel Ensemble
    NUM_STREAMS = 5

    # Stream Configurations (Heterogeneous)
    # Streams 1 & 2: Standard Funnel, Dropout 0.20
    # Streams 3 & 4: Wide Funnel, Dropout 0.25
    # Stream 5: Standard Funnel, Dropout 0.30
    STREAM_CONFIGS = [
        {"hidden_layers": [512, 256, 128], "dropout": 0.20},  # Stream 1
        {"hidden_layers": [512, 256, 128], "dropout": 0.20},  # Stream 2
        {"hidden_layers": [1024, 512, 256], "dropout": 0.25},  # Stream 3
        {"hidden_layers": [1024, 512, 256], "dropout": 0.25},  # Stream 4
        {"hidden_layers": [512, 256, 128], "dropout": 0.30},  # Stream 5
    ]

    # Activation function
    ACTIVATION = "ReLU"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 50

    # Optimizer: AdamW
    OPTIMIZER_NAME = "AdamW"
    WEIGHT_DECAY = 2e-5

    # Scheduler: OneCycleLR
    SCHEDULER_NAME = "OneCycleLR"
    MAX_LR = 1e-2
    PCT_START = 0.3  # Standard OneCycle parameter
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 10000.0

    # Early Stopping
    PATIENCE = 10

    @classmethod
    def setup(cls):
        """Creates necessary directories for outputs and caching."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
