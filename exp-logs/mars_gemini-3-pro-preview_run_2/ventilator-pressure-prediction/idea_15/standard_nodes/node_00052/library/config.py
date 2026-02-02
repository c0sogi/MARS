import os
import torch


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    CACHE_DIR = WORKING_DIR  # For cached processed data (parquet/npy)
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # ==========================================
    # 3. Data & Feature Engineering
    # ==========================================
    # Feature Flags
    USE_ROBUST_SCALER = True  # Handles u_in outliers better
    USE_LAG_FEATURES = True
    USE_DIFF_FEATURES = True  # 1st and 2nd derivatives
    USE_PHYSICS_TERMS = True  # R*u_in, u_in_cumsum/C

    # Sequence Properties
    MAX_SEQ_LEN = 80  # Standard breath length in this dataset

    # ==========================================
    # 4. Model Architecture (DFLB-BiLSTM)
    # ==========================================
    # Deep Front-End (Gated Residual MLP)
    FRONT_END_LAYERS = 3
    FRONT_END_DIM = 256

    # Bottleneck
    BOTTLENECK_DIM = 128  # Compressed context injection

    # Recurrent Backbone
    LSTM_DIM = 512
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Injection Strategy
    DEEP_INJECTION = True  # Inject payload into every LSTM layer

    # Head
    HEAD_DROPOUT = 0.0

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    EPOCHS = 200
    BATCH_SIZE = 512  # A100 40GB can handle large batches

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = 200  # Matches EPOCHS for stretched horizon
    SCHEDULER_ETA_MIN = 1e-6

    # Loss Weights (Weighted L1)
    # Focus on inspiratory phase (u_out=0), less on expiratory (u_out=1)
    LOSS_WEIGHT_INSPIRATORY = 1.0
    LOSS_WEIGHT_EXPIRATORY = 0.1

    def __str__(self):
        """Prints the configuration."""
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]
        config_str = "==== Configuration ====\n"
        for attr in attributes:
            config_str += f"{attr}: {getattr(self, attr)}\n"
        config_str += "======================="
        return config_str
