import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Three-Stage 'Generate-and-Discriminate' Pipeline.

    Stages:
    1. Locator (DeBERTa-v3-Base): Identifies Top-K gap locations.
    2. In-Filler (RoBERTa-Large): Predicts candidate words for gaps.
    3. Verifier (DeBERTa-v3-Large): Re-ranks candidates based on global coherence.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging, False for full training

    # Hardware
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    MAX_LENGTH = 128  # Efficient length for sentence-level tasks

    # Dataset Sampling
    # Full training uses 2 million samples as per strategy
    TRAIN_SIZE = 2_000_000
    VAL_SIZE = 100_000

    # Debug sizes
    DEBUG_TRAIN_SIZE = 10_000
    DEBUG_VAL_SIZE = 2_000

    @classmethod
    def get_train_size(cls):
        return cls.DEBUG_TRAIN_SIZE if cls.DEBUG else cls.TRAIN_SIZE

    @classmethod
    def get_val_size(cls):
        return cls.DEBUG_VAL_SIZE if cls.DEBUG else cls.VAL_SIZE

    # ==========================================
    # Stage 1: Locator (Token Classification)
    # ==========================================
    LOCATOR_MODEL_NAME = "microsoft/deberta-v3-base"
    LOCATOR_BATCH_SIZE = 32
    LOCATOR_LR = 2e-5
    LOCATOR_EPOCHS = 3
    LOCATOR_TOP_K = 5  # Beam width: Number of candidate locations to propose
    LOCATOR_CKPT_PATH = os.path.join(WORKING_DIR, "best_locator.pth")

    # ==========================================
    # Stage 2: In-Filler (Masked Language Model)
    # ==========================================
    INFILLER_MODEL_NAME = "roberta-large"
    INFILLER_BATCH_SIZE = 32
    INFILLER_LR = 1e-5
    INFILLER_EPOCHS = 3
    INFILLER_CKPT_PATH = os.path.join(WORKING_DIR, "best_infiller.pth")

    # ==========================================
    # Stage 3: Verifier (Sequence Classification)
    # ==========================================
    VERIFIER_MODEL_NAME = "microsoft/deberta-v3-large"
    VERIFIER_BATCH_SIZE = 16  # Larger model, smaller batch
    VERIFIER_LR = 1e-5
    VERIFIER_EPOCHS = 2
    VERIFIER_LAMBDA = 2.0  # Weight for the verifier score in final ranking
    VERIFIER_LABEL_SMOOTHING = 0.1
    VERIFIER_CKPT_PATH = os.path.join(WORKING_DIR, "best_verifier.pth")

    # ==========================================
    # Utilities
    # ==========================================
    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
