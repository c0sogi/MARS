import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data (Idea 3 specific)
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata files (contain IDs and labels, point to source files)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Raw data files
    TRAIN_RAW = os.path.join(INPUT_DIR, "train.csv")
    TEST_RAW = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "roberta-base"
    MAX_LEN = 300
    NUM_LABELS = 6
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Hidden size for RoBERTa-base is 768
    HIDDEN_SIZE = 768

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64

    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    DROPOUT = 0.1

    # Early Stopping
    PATIENCE = 2

    # ==========================================
    # Hardware & Performance
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
