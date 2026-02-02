import os
import torch


class Config:
    """
    Central configuration for the Text-Augmented DistilRoBERTa Dual-Encoder experiment.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Input Files (using generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "distilroberta-base"
    NUM_TARGETS = 30

    # Wide-Bottleneck Fusion settings
    BOTTLENECK_DIM = 768
    DROPOUT = 0.1

    # ==========================================
    # Text Processing
    # ==========================================
    MAX_LEN = 512  # Maximum sequence length for the tokenizer
    TOKENIZER_NAME = MODEL_NAME

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 5
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32

    # Differential Learning Rates
    LEARNING_RATE_BACKBONE = 2e-5
    LEARNING_RATE_HEAD = 1e-3

    # Optimizer settings
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Scheduler
    WARMUP_RATIO = 0.1

    # ==========================================
    # Hardware & Performance
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
