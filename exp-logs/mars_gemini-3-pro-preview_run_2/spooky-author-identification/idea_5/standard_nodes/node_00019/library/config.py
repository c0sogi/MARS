import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Author Identification pipeline.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Toggle for debugging with smaller dataset
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Model Architectures
    # ==========================================
    # Branch A: Syntactic-Semantic Transformer
    MODEL_DEBERTA = "microsoft/deberta-v3-large"

    # Branch B: Global Context Transformer
    MODEL_ROBERTA = "roberta-large"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    MAX_LEN = 85
    BATCH_SIZE = 8  # Optimized for A100 40GB with Large models
    EPOCHS = 10  # Max epochs, managed by Early Stopping
    PATIENCE = 1  # Aggressive early stopping to prevent overfitting
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 1

    # ==========================================
    # Classical Model Hyperparameters
    # ==========================================
    TFIDF_MIN_DF = 2
    SVD_COMPONENTS = 100

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    @classmethod
    def create_dirs(cls):
        """
        Ensures that the working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=42):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize environment immediately upon import
Config.create_dirs()
seed_everything(Config.SEED)
