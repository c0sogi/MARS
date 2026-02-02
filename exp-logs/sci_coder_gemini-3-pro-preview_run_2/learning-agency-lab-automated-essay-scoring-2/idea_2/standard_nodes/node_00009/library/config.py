import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Essay Scoring project.
    Acts as the single source of truth for paths, hyperparameters, and constants.
    """

    # =============================================================================
    # PATHS
    # =============================================================================
    # Input Metadata (Stratified Splits)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Artifacts
    # Idea 2 represents the Deep Learning Regression approach
    WORKING_DIR = "./working/idea_2"

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories immediately upon config load
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =============================================================================
    # MODEL HYPERPARAMETERS
    # =============================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"

    # Extended context length to capture full essay content (EDA showed max ~1300 words)
    # A100 40GB can handle 1024 tokens with appropriate batch size
    MAX_LENGTH = 1024

    # Regression output (1 scalar value)
    NUM_LABELS = 1

    # =============================================================================
    # TRAINING HYPERPARAMETERS
    # =============================================================================
    SEED = 42

    # Training duration
    EPOCHS = 4

    # Batch Size Strategy for A100 40GB
    # 1024 tokens is memory intensive.
    # Batch size 4 fits comfortably with mixed precision and gradient checkpointing.
    BATCH_SIZE = 4
    GRAD_ACCUMULATION_STEPS = 4  # Effective Batch Size = 16

    # Optimizer settings
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # =============================================================================
    # COMPUTE
    # =============================================================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # POST-PROCESSING
    # =============================================================================
    # Whether to use Nelder-Mead to optimize classification thresholds on validation set
    OPTIMIZE_THRESHOLDS = True


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across all libraries.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
