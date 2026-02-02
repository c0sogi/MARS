import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the Hybrid-Stem Direct-Access Recurrent Network (HS-DARN).
    Defines global hyperparameters, file paths, and reproducibility settings.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_56"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Configuration
    CACHE_KEY = "hs_darn_v1"

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target Columns
    # 'TARGET_COLS' includes all 5 conditions required for the submission format.
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # 'SCORED_COLS' includes only the 3 conditions used for the competition metric.
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Vocabularies
    VOCAB_SIZE_SEQ = 4  # Bases: A, G, U, C
    VOCAB_SIZE_STRUCT = 3  # Structure: ., (, )
    VOCAB_SIZE_LOOP = 7  # Loop Types: S, M, I, B, H, E, X

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    LATENT_DIM = 64
    FEEDBACK_DIM = 32
    HIDDEN_DIM = 64  # For RNN / Global Aggregation
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch Size strictly set to 16 as per Lesson 00129 (Small Batch Regime)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    NUM_WORKERS = 2

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
