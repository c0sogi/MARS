import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # 1. Paths
    # ==========================================
    # Input Metadata (Generated in previous steps)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Working Directory for Cache and Checkpoints
    WORKING_DIR = "./working/idea_4/"
    os.makedirs(WORKING_DIR, exist_ok=True)

    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_checkpoint")

    # Output Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Model Hyperparameters
    # ==========================================
    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    MAX_GRAD_NORM = 1.0

    # Subsampling strategy for PLAIN class to handle imbalance
    # Keep 100% of non-PLAIN sentences, keep X% of purely PLAIN sentences
    PLAIN_SUBSAMPLE_RATIO = 0.1

    # ==========================================
    # 3. Label Mappings
    # ==========================================
    # Comprehensive list of classes found in Google Text Normalization datasets
    LABELS = [
        "PLAIN",
        "PUNCT",
        "DATE",
        "LETTERS",
        "CARDINAL",
        "VERBATIM",
        "MEASURE",
        "ORDINAL",
        "DECIMAL",
        "MONEY",
        "DIGIT",
        "ELECTRONIC",
        "TELEPHONE",
        "TIME",
        "ADDRESS",
        "FRACTION",
    ]

    # Create mappings
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}
    NUM_LABELS = len(LABELS)

    # ==========================================
    # 4. Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    SEED = 42


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
