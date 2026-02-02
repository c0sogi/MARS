import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 5000

    # ==========================================
    # Data Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Source Data Files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "transformer_model.bin")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "distilbert-base-uncased"
    # 256 covers the vast majority of comments (mean char len ~300)
    # while fitting comfortably in memory with larger batches.
    MAX_LEN = 256

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # A100 40GB can handle larger batches for DistilBERT
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 1

    # ==========================================
    # Bias Mitigation & Loss
    # ==========================================
    # Columns used for identifying subgroups
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

    # Target column
    TARGET_COL = "target"
    TEXT_COL = "comment_text"

    # Weighting for Bias Loss
    # We apply higher sample weights to:
    # 1. Toxic examples mentioning an identity (BNSP trap)
    # 2. Non-toxic examples mentioning an identity (BPSN trap)
    BIAS_LOSS_WEIGHT = 5.0

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Print configuration status
        print(f"Configuration Loaded.")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Batch Size: {cls.TRAIN_BATCH_SIZE}")
        print(f"  Bias Weight: {cls.BIAS_LOSS_WEIGHT}")
