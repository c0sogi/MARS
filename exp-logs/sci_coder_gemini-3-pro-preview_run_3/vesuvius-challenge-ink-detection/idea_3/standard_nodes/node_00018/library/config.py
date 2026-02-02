import os
import torch


class Config:
    """
    Central configuration for the Parallel-Scale Dilated Network (PSDN) project.
    """

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to Idea 3 (PSDN) to avoid conflicts
    WORKING_DIR = "./working/idea_3"

    # Final submission file location (Home directory as per competition format)
    SUBMISSION_FILE = "submission.csv"

    # -------------------------------------------------------------------------
    # Data Specification
    # -------------------------------------------------------------------------
    # Standardized Z-depth of the 3D surface volumes
    Z_DIM = 65

    # Patch size for training and inference
    # 128x128 provides sufficient spatial context for dilation rate 8
    PATCH_SIZE = 128

    # Stride for sliding window inference (overlap reduces edge artifacts)
    INFERENCE_STRIDE = 64

    # -------------------------------------------------------------------------
    # Model Architecture: Parallel-Scale Dilated Network (PSDN)
    # -------------------------------------------------------------------------
    # Lean channel width to prevent overfitting on limited data
    MODEL_CHANNELS = 32

    # Parallel dilation rates to capture multi-scale context simultaneously
    # Rate 1: High-freq details; Rate 8: Global fiber context
    DILATION_RATES = [1, 2, 4, 8]

    # Dropout for regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3

    # Training duration
    NUM_EPOCHS = 15

    # Since we sample patches randomly, we define an epoch by a fixed number of steps
    # Increased to ~16k patches per epoch to stabilize gradients (Cite solution_lesson_node_00003)
    STEPS_PER_EPOCH = 1000

    # Early stopping patience
    PATIENCE = 5

    # -------------------------------------------------------------------------
    # Validation & Evaluation
    # -------------------------------------------------------------------------
    # Fixed, large validation set size to reduce metric volatility (Cite solution_lesson_node_00010)
    VAL_SAMPLE_SIZE = 4000

    # Threshold tuning range for F0.5 optimization
    THRESHOLD_START = 0.2
    THRESHOLD_END = 0.8
    THRESHOLD_STEP = 0.05

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # Compute resources
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def ensure_directories(cls):
        """
        Creates the necessary working directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically ensure directories exist upon import
Config.ensure_directories()
