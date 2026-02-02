import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Essay Scoring pipeline.
    Implements parameters for the Quad-Branch Heterogeneous Stacking Network.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Data paths (using metadata as requested)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # =========================================================================
    # Branch 1: Deep Semantic (DeBERTa-v3-large)
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 4  # Adjusted for A100 40GB with 1024 seq len
    VALID_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4  # Effective batch size = 4 * 4 = 16
    EPOCHS = 4
    LEARNING_RATE = 1e-5  # Uniform learning rate
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1000
    SCHEDULER = "cosine"
    WARMUP_RATIO = 0.1

    # Loss
    LOSS_FN = "SmoothL1Loss"

    # Adversarial Weight Perturbation (AWP)
    USE_AWP = True
    AWP_START_EPOCH = 2
    AWP_LR = 1e-4
    AWP_EPS = 1e-4

    # =========================================================================
    # Branch 2: Lexical (Word N-grams)
    # =========================================================================
    WORD_NGRAM_RANGE = (1, 3)
    WORD_MIN_DF = 3
    RIDGE_ALPHA_WORD = 1.0

    # =========================================================================
    # Branch 3: Morphological (Char N-grams)
    # =========================================================================
    CHAR_NGRAM_RANGE = (3, 5)
    CHAR_MIN_DF = 50  # Higher threshold for chars to reduce noise
    RIDGE_ALPHA_CHAR = 1.0

    # =========================================================================
    # Branch 4: Structural (Linguistic Features)
    # =========================================================================
    # List of features to be computed (conceptual list for feature engineering module)
    STRUCTURAL_FEATURES = [
        "word_count",
        "char_count",
        "sentence_count",
        "avg_word_len",
        "avg_sentence_len",
        "unique_word_count",
        "unique_word_ratio",
        "punctuation_count",
        "spelling_error_count",
    ]

    # =========================================================================
    # Meta-Learner (Stacking)
    # =========================================================================
    N_FOLDS = 5
    META_MODEL_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

        # Deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Suppress parallelism warnings for tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Config setup complete. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
