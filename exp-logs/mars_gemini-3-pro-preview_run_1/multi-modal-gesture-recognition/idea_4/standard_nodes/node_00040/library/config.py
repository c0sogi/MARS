import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Context-Gated Residual GRU (CGR-GRU) experiment.
    Centralizes all hyperparameters, file paths, and setup utilities.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 4
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Ensure critical working directories exist upon import
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    SEED = 42
    FPS = 20
    AUDIO_SR = 16000

    # Classes: 20 Gestures (1-20) + 1 Background (0)
    NUM_CLASSES = 21

    # Input Dimensions
    N_JOINTS = 20
    CHANNELS_PER_JOINT = 3  # (x, y, z) relative coordinates
    SKELETON_INPUT_DIM = N_JOINTS * CHANNELS_PER_JOINT  # 60

    N_MFCC = 13
    AUDIO_INPUT_DIM = N_MFCC

    # ==========================================
    # Model Architecture
    # ==========================================
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    NUM_RNN_LAYERS = 2  # For Residual BiGRU stack

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 60

    # Optimization Strategies
    LABEL_SMOOTHING = 0.1
    BG_CLASS_WEIGHT = 0.7
    WEIGHT_DECAY = 0.05
    EARLY_STOPPING_PATIENCE = 20

    # Debugging: Set to int (e.g., 100) to limit dataset size
    DEBUG_SUBSET_SIZE = None

    # ==========================================
    # Utilities
    # ==========================================
    @staticmethod
    def get_device():
        """Returns the appropriate torch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=None):
        """Sets fixed random seeds for reproducibility."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
