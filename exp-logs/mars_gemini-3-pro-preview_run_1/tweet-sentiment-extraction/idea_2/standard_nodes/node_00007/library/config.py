import os
import random
import numpy as np
import torch


class Config:
    # Meta
    SEED = 42
    DEBUG = False  # Set to True for quick debugging runs
    DEBUG_SAMPLE_SIZE = 100  # Number of rows to use if DEBUG is True

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Artifacts
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "roberta_model.bin")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Model Architecture
    ROBERTA_PATH = "roberta-base"
    TOKENIZER_PATH = "roberta-base"
    HIDDEN_SIZE = 768
    N_LAST_HIDDEN = (
        4  # Number of hidden layers to aggregate (Multi-Layer Feature Aggregation)
    )
    DROPOUT = 0.1

    # Tokenization
    MAX_LEN = 96  # Tweets are short, 96 covers most cases comfortably

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Strategy Specifics
    FILTER_NEUTRAL_TRAIN = True  # Exclude neutral tweets from training
    LABEL_SMOOTHING = 0.1  # Parameter for Gaussian Label Smoothing or Soft Labeling


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
