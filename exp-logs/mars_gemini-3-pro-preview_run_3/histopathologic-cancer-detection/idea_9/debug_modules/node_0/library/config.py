import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Converged Heterogeneous Stacking Strategy.
    """

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # --- Data Paths ---
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # --- Image Preprocessing ---
    # Original patch size is 96x96
    ORIGINAL_SIZE = 96
    # Center crop to 64x64 to focus on the 32x32 ROI + 16px context buffer
    CROP_SIZE = 64

    # --- Model Architecture ---
    # Heterogeneous backbones for stacking
    # 'convnext_tiny': Hierarchical, Transformer-inspired
    # 'densenet121': Dense connectivity, distinct feature reuse
    MODELS = ["convnext_tiny", "densenet121"]

    # --- Training Hyperparameters ---
    SEED = 42
    NUM_FOLDS = 5
    # Extended to 30 epochs to ensure full convergence
    EPOCHS = 30
    # Maximize batch size for A100 (40GB) to stabilize BN and gradients
    BATCH_SIZE = 512
    NUM_WORKERS = 4

    # --- Optimization ---
    # Learning rate for cosine annealing
    LR = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # --- Meta-Learner (XGBoost) ---
    # Parameters for the Level-1 stacker
    META_MODEL_PARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility across libraries.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
