import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Essay Scoring task.
    Handles hyperparameters, file paths, and environment setup.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_9"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{EXP_NAME}"
    SUBMISSION_DIR = "./submission"

    # Input files (using generated metadata)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache paths (constructed dynamically in processing modules, but base is here)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    OUTPUT_LOG_DIR = os.path.join(WORKING_DIR, "output")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_PATH = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1024  # Max token length for the transformer
    INFERENCE_MAX_LENGTH = 1536  # Slightly larger context for inference if needed

    # Architecture specifics
    HIDDEN_SIZE = 1024  # Deberta-v3-large hidden size
    NUM_LABELS = 1  # For Regression head
    NUM_ORDINAL_LABELS = 5  # For Ordinal Classification head (P(>1), P(>2)... P(>5))

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 4

    # Batch sizes (tuned for A100 40GB)
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4

    # Optimization
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1000

    # Layer-wise Learning Rate Decay (LLRD)
    LLRD_DECAY = 0.9

    # Adversarial Weight Perturbation (AWP)
    USE_AWP = True
    AWP_START_EPOCH = 1  # Start AWP after the first epoch
    AWP_LR = 1e-4
    AWP_EPS = 1e-2

    # Scheduler
    WARMUP_RATIO = 0.1

    # Hardware
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Stacking / Meta-Model Hyperparameters
    # =========================================================================
    LGBM_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.005,
        "metric": "rmse",
        "random_state": SEED,
        "boosting_type": "gbdt",
        "objective": "regression",
        "subsample": 0.8,
        "subsample_freq": 1,
        "feature_fraction": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1,
    }

    @classmethod
    def setup_environment(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        for directory in [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.OUTPUT_LOG_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

        # Enforce deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prevent tokenizer parallelism deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize environment immediately upon import
Config.setup_environment()
