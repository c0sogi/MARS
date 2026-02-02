import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # Experiment Identity
    EXPERIMENT_NAME = "idea_23"

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Model Architecture: Wide-Stream BiGRU with Fixed Geometric Bias
    HIDDEN_DIM = 384  # Residual stream width
    NUM_LAYERS = 6  # Number of residual blocks
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 32  # Cite Lesson 00056: Maintain gradient update budget
    LR = 1e-3
    EPOCHS = 20
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Vocabularies
    # Atomic Sequence: A, G, C, U
    NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    # Loop Context: S, M, I, B, H, E, X
    LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Targets
    # Only training on the 3 scored columns as per strategy
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    SEED = 42

    @classmethod
    def setup(cls):
        """Ensures working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Configuration initialized. Working dir: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
