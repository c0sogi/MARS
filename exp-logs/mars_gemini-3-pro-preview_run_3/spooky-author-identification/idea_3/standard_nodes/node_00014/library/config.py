import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Compute Resources
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of workers for data loading

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data (Metadata)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Neural Branch Configuration
    # =========================================================================
    # Model Architectures
    MODEL_DEBERTA = "microsoft/deberta-v3-large"
    MODEL_ROBERTA = "roberta-large"

    # Tokenization
    MAX_LEN = 128  # Sufficient for the majority of sentences (mean char len ~148)

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 2

    # =========================================================================
    # Statistical Branch Configuration
    # =========================================================================
    # TF-IDF Parameters
    TFIDF_MAX_FEATURES = 50000
    WORD_NGRAM_RANGE = (1, 2)
    CHAR_NGRAM_RANGE = (3, 5)

    # =========================================================================
    # Meta-Learner Configuration
    # =========================================================================
    META_C = 1.0
    META_SOLVER = "lbfgs"

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed)
