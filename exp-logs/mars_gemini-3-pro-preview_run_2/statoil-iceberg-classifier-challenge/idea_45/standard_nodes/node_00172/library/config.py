import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Context-Gated Wide-Body Network (CG-WBN).
    Contains paths, hyperparameters, and reproducibility settings.
    """

    # ==========================================
    # 1. GLOBAL SETTINGS & REPRODUCIBILITY
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    MAX_SAMPLES = (
        None  # If DEBUG is True, limits dataset to this many samples (e.g., 100)
    )

    # ==========================================
    # 2. DATA PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Stratified Splits)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # 3. OUTPUT PATHS & ARTIFACTS
    # ==========================================
    # Specific working directory for this experiment (Idea 46)
    IDEA_ID = "idea_46"
    WORK_DIR = os.path.join("./working", IDEA_ID)

    # Cache file for processed tensors (images and metadata)
    CACHE_FILE = os.path.join(WORK_DIR, "processed_data.npz")

    # Checkpoint storage pattern
    MODEL_CHECKPOINT_PATTERN = os.path.join(WORK_DIR, "model_fold_{fold}.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 4. DATA SPECIFICATIONS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Channels: Band 1 (HH), Band 2 (HV), Mean ((HH+HV)/2)
    NUM_CHANNELS = 3

    # ==========================================
    # 5. MODEL HYPERPARAMETERS
    # ==========================================
    # "Low and Slow" Optimization Strategy
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 100  # Generous limit to allow slow convergence
    PATIENCE = 15  # Early stopping patience

    # Regularization
    DROPOUT_RATE = 0.5

    # Architecture Specifics
    BACKBONE_FILTERS = 128  # Sustained width
    READOUT_DIM = 1024  # Dimension before gating

    # ==========================================
    # 6. TRAINING CONFIGURATION
    # ==========================================
    NUM_FOLDS = 5
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
