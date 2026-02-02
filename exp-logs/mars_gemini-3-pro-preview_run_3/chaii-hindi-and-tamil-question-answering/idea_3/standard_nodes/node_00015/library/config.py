import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering Task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Data (Metadata)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    OUTPUT_DIR = WORKING_DIR
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (Parquet/NPY)
    # Used for deterministic data processing caching
    TAPT_CACHE_DIR = os.path.join(WORKING_DIR, "tapt_cache")
    QA_CACHE_DIR = os.path.join(WORKING_DIR, "qa_cache")

    # Model Checkpoints
    TAPT_MODEL_PATH = os.path.join(WORKING_DIR, "tapt_model_finetuned")
    QA_MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "qa_models")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "xlm-roberta-base"
    NUM_LABELS = 3  # Labels: O (0), B-ANS (1), I-ANS (2)

    # =========================================================================
    # Data Processing
    # =========================================================================
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # =========================================================================
    # Training Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Can be toggled to run on a subset

    # TAPT (Task-Adaptive Pretraining) Settings
    TAPT_EPOCHS = 5
    TAPT_BATCH_SIZE = 8
    TAPT_LEARNING_RATE = 2e-5
    TAPT_MLM_PROB = 0.15

    # QA Fine-tuning Settings
    N_FOLDS = 1
    EPOCHS = 10
    SEEDS = [42, 43, 44]
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3  # For Early Stopping

    # Hardware
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Create necessary directories and set random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.TAPT_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.QA_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.QA_MODEL_OUTPUT_DIR, exist_ok=True)

        # Set Seed
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
