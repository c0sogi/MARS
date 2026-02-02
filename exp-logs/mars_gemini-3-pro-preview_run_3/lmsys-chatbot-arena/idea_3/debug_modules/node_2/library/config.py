import os
import torch


class Config:
    """
    Centralized configuration for the Longformer-based Chatbot Arena prediction task.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SIZE = 1000  # Number of rows to use when DEBUG is True

    # ==========================================
    # Directory & File Paths
    # ==========================================
    # Metadata directories (Input)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for artifacts (checkpoints, cache)
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache directory for processed datasets
    CACHE_DIR = WORKING_DIR

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "allenai/longformer-base-4096"
    NUM_LABELS = 3

    # Input Sequence Length
    # Longformer supports up to 4096, but we use 2048 to optimize throughput on A100
    MAX_LENGTH = 2048

    # Dropout
    HIDDEN_DROPOUT_PROB = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 3

    # Batch size per GPU
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4

    # Gradient Accumulation to simulate larger effective batch size
    # Effective Batch Size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    GRADIENT_ACCUMULATION_STEPS = 4

    # Optimizer settings
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Mixed Precision
    USE_FP16 = True

    # ==========================================
    # Data Loading & Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Targets
    # ==========================================
    TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
