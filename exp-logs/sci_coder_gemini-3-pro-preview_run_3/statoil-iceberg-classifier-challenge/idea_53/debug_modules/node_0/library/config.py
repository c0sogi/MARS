import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Experiment Identity
    # ==========================================
    PROJECT_NAME = "Iceberg_Classifier_DIDP_CNN"
    IDEA_ID = "idea_53"
    SEED = 42

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = os.path.join("./working", IDEA_ID)

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Paths (referenced by metadata, but good to have)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Synthetic Average
    NUM_CLASSES = 1  # Binary: 0=Ship, 1=Iceberg

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Model Architecture (DIDP-CNN)
    # ==========================================
    # Backbone: Plain CNN (4 blocks)
    # Width Strategy: 64 -> 128 -> 128 -> 128 (Early Expansion)
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Readout: 256 dimensions (Decoupled: 64 Peak + 64 Shadow per stage for stages 3 & 4)
    # 64*4 = 256
    READOUT_DIM = 256

    # Regularization
    DROPOUT_RATE = 0.5
    LEAKY_RELU_SLOPE = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    NUM_EPOCHS = 75
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-2  # L2 Regularization (Standard for AdamW)
    PATIENCE = 12  # Early Stopping

    # Hardware
    NUM_WORKERS = 8  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories setup complete at {cls.WORKING_DIR}")


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
