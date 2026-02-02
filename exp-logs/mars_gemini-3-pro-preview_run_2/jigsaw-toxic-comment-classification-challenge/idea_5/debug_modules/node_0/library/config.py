import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging: Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Input files (using metadata splits as requested)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output directories
    OUTPUT_DIR = WORKING_DIR
    TFIDF_CACHE_DIR = os.path.join(WORKING_DIR, "tfidf_cache")

    # Ensure working directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TFIDF_CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    LABEL_COLS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    NUM_LABELS = 6
    MAX_LEN = 256  # Extended context window for large models

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Branch A: DeBERTa-v3-Large
    MODEL_A_NAME = "microsoft/deberta-v3-large"

    # Branch B: RoBERTa-Large
    MODEL_B_NAME = "roberta-large"

    # Branch C: Linear Model N-grams
    LINEAR_WORD_NGRAMS = (1, 2)
    LINEAR_CHAR_NGRAMS = (2, 6)
    LINEAR_MAX_FEATURES = 100000

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 3

    # Batch size configuration for A100 (40GB)
    # Large models consume significant VRAM.
    # Batch size 8 * Accumulation 4 = Effective Batch Size 32
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4

    # Optimization
    LR = 1e-5
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Scheduler
    SCHEDULER = "cosine"
    WARMUP_RATIO = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    LLRD_DECAY = 0.95

    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    # =========================================================================
    USE_AWP = True
    AWP_START_EPOCH = 1  # Start AWP after the first epoch (0-indexed 1 is epoch 2)
    AWP_EPS = 1e-2
    AWP_LR = 1e-4

    # =========================================================================
    # Utility Methods
    # =========================================================================
    @classmethod
    def set_seed(cls):
        """Sets the random seed for reproducibility."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def get_run_name(cls, model_name):
        """Generates a standardized run name for saving artifacts."""
        clean_name = model_name.split("/")[-1].replace("-", "_")
        return f"model_{clean_name}"
