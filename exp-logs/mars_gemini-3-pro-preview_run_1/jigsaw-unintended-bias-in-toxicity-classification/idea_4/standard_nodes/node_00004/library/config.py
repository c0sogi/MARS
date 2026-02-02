import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DATA_LIMIT = 5000 if DEBUG else None  # Number of rows to load if DEBUG is True
    NUM_WORKERS = 4

    # ==========================================
    # Paths
    # ==========================================
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output / Working Directories
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "multitask_transformer.bin")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN = 256

    # Primary Head (Toxicity)
    NUM_LABELS = 1

    # Auxiliary Head (Identity)
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]
    NUM_IDENTITY_LABELS = len(IDENTITY_COLUMNS)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 0  # Can be calculated dynamically based on len(train_loader)
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 2

    # ==========================================
    # Multi-Task & Bias Weighting Strategies
    # ==========================================
    # Weight for the auxiliary identity loss (lambda)
    # L_total = L_toxicity + (AUX_LOSS_WEIGHT * L_identity)
    AUX_LOSS_WEIGHT = 0.25

    # Sample Weighting Factors
    # We upweight specific subgroups to penalize bias:
    # 1. Toxic comments mentioning identity
    # 2. Non-toxic comments mentioning identity
    IDENTITY_WEIGHT_FACTOR = 5.0
    BACKGROUND_WEIGHT_FACTOR = 1.0
