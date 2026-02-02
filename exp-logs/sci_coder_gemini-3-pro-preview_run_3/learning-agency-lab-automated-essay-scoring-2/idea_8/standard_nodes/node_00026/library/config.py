import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the Essay Scoring Task.
    Implements settings for 'Full-Scale Robust K-Fold Ensemble with Adversarial Training'.
    """

    # =========================================================================
    # General Experiment Settings
    # =========================================================================
    EXP_NAME = "idea_8"
    DEBUG = False  # Set to True for fast debugging on small subsets
    SEED = 42
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # Data Paths (Using pre-generated stratified metadata)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Storage
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Input Processing Strategy
    MAX_LENGTH = 512  # Maximum token length per window
    WINDOW_SIZE = 512  # Sliding window size
    WINDOW_STRIDE = 128  # Stride for sliding window

    # Pooling Mechanism
    POOLING_TYPE = "attention"  # Attention pooling on the final hidden layer

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 4

    # Batch Sizes (Optimized for A100 40GB)
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4

    # Optimization
    LEARNING_RATE = 1e-5  # Base learning rate for backbone
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 10.0  # Relaxed clipping for AWP
    WARMUP_RATIO = 0.1
    SCHEDULER_TYPE = "cosine"

    # Layer-wise Learning Rate Decay (LLRD)
    LLRD_DECAY = 0.9

    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    # =========================================================================
    USE_AWP = True
    AWP_START_EPOCH = 1  # Enable AWP after the 1st epoch (0-indexed)
    AWP_EPS = 1e-2  # Perturbation magnitude
    AWP_LR = 1e-4  # AWP learning rate

    # =========================================================================
    # Stacking (LightGBM)
    # =========================================================================
    # Head trained on concatenated OOF Embeddings + Explicit Meta-Features
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.005,
        "metric": "rmse",
        "objective": "regression",
        "boosting_type": "gbdt",
        "device": "cpu",
        "random_state": SEED,
        "force_col_wise": True,
        "feature_fraction": 0.7,  # Subsample features to force usage of embeddings
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_leaves": 31,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "early_stopping_rounds": 100,
    }

    # =========================================================================
    # Hardware & Environment
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Initialize the experiment environment:
        1. Create necessary directories for working, cache, checkpoints, and submission.
        2. Set random seeds for reproducibility across random, numpy, and torch.
        """
        # Create directories
        for d in [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

        # Enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Suppress tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Config Setup Complete. Working Directory: {cls.WORKING_DIR}")
