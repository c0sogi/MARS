import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration module for the Depth-Specialist SegFormer Ensemble (DS-SegFormer).
    Defines global hyperparameters, specialist ranges, and data processing constants.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and saving model checkpoints
    WORKING_DIR = "./working/idea_21"
    # Path for the final submission file
    SUBMISSION_PATH = "submission.csv"

    # --- Reproducibility ---
    SEED = 42

    # --- Data Generation Parameters ---
    TILE_SIZE = 512
    # Stride for generating training patches (non-overlapping)
    TRAIN_STRIDE = 512

    # --- Specialist Z-Ranges ---
    # Defines the specific Z-depth windows for each specialist model.
    # Each range covers exactly 24 slices.
    SPECIALIST_RANGES = {
        "A": (16, 40),  # High Specialist
        "B": (20, 44),  # Center Specialist
        "C": (24, 48),  # Low Specialist
    }

    # --- Slab Projection Strategy ---
    # Parameters for mapping the 24-slice Z-window to 3 input channels.
    # We use an "Overlapping Thick Slab" strategy.
    # Channel 0: slices [0:12] relative to start
    # Channel 1: slices [6:18] relative to start
    # Channel 2: slices [12:24] relative to start
    SLAB_THICKNESS = 12
    SLAB_OVERLAP = 6
    SLAB_STRIDE = 6  # Derived: thickness - overlap

    # --- Model Architecture ---
    ENCODER_NAME = "mit_b2"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1

    # --- Training Hyperparameters ---
    # Strict batch size to prevent underfitting on small dataset
    BATCH_SIZE = 8
    # Conservative learning rate for stability
    LEARNING_RATE = 6e-5
    # Number of training epochs
    EPOCHS = 15
    # Number of data loading workers
    NUM_WORKERS = 2

    # --- Validation & Inference ---
    # Minimum F0.5 score required to save a model checkpoint
    VALID_THRESHOLD = 0.55

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.

        Args:
            seed (int): The random seed to use.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories
        and setting random seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        cls.set_seed(cls.SEED)
