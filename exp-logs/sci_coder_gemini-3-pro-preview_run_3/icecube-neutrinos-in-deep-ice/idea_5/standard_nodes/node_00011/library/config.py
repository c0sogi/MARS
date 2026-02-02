import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")
    SENSOR_GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model.pth")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    # Fixed number of pulses to sample per event (Hybrid Sampling: Charge + Time)
    NUM_PULSES = 196

    # Debugging / Development
    # Set to True to use a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Dynamic EdgeConv
    K_NEIGHBORS = 20
    EMBED_DIM = 128

    # Transformer Aggregator
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # A100 40GB allows for large batch sizes.
    # 95M events is very large, so large batch size helps throughput.
    BATCH_SIZE = 1024
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 3

    # ==========================================
    # Hardware & Reproducibility
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Available vCPUs
    SEED = 42


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    """
    Ensures that the working and submission directories exist.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
