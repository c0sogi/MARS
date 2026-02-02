import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the settings for the Channel-Weighted Wide-Stream Residual BiGRU strategy.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    PROJECT_NAME = "idea_29"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Targets to be predicted and scored
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies for Atomic Sequence and Loop Type
    # Atomic Sequence: A, G, C, U
    TOKEN_VOCAB = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Predicted Loop Type: S (Stem), M (Multiloop), I (Internal), B (Bulge),
    # H (Hairpin), E (Dangling End), X (External)
    LOOP_VOCAB = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # High-Dimensional Embeddings for input channels
    EMBED_DIM = 128

    # Wide Stream Capacity
    HIDDEN_DIM = 512

    # Depth
    NUM_LAYERS = 6

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 1e-3
    # Low weight decay specifically for RNN stability (Lesson 00070)
    WEIGHT_DECAY = 1e-4

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
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
    os.environ["PYTHONHASHSEED"] = str(seed)
