import os
import torch
import random
import numpy as np


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


class Config:
    """
    Global configuration for the Sparse Retrieval-Augmented Sequence Labeling pipeline.
    """

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    # Input Metadata Paths (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (e.g., processed datasets)
    WORKING_DIR = "./working/idea_1"

    # Output directory for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Using a lightweight multilingual model suitable for Hindi and Tamil
    MODEL_NAME = "distilbert-base-multilingual-cased"

    # Maximum sequence length for the tokenizer.
    # Increased to 384 to handle larger context chunks with sliding window.
    MAX_LEN = 384
    DOC_STRIDE = 128

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Early Stopping to prevent overfitting
    EARLY_STOPPING_PATIENCE = 3

    # --------------------------------------------------------------------------
    # Label Configuration (BIO Tagging)
    # --------------------------------------------------------------------------
    # 0: Outside, 1: Beginning of Answer, 2: Inside Answer
    LABELS_TO_IDS = {"O": 0, "B-ANS": 1, "I-ANS": 2}
    IDS_TO_LABELS = {0: "O", 1: "B-ANS", 2: "I-ANS"}
    NUM_LABELS = 3

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    @staticmethod
    def setup():
        """
        Initializes the environment by creating necessary directories
        and setting random seeds.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        set_seed(Config.SEED)
