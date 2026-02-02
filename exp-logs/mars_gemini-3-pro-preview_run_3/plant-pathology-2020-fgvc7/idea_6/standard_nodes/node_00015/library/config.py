import os
import torch
import numpy as np
import random


class Config:
    """
    Central configuration for the Apple Disease Detection task.
    Implements the Dual-Backbone Hierarchical Ensemble strategy settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for specific idea iteration
    WORK_DIR = "./working/idea_7"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone 1: Multi-Level EfficientNet-B4
    MODEL_EFFNET_NAME = "tf_efficientnet_b4_ns"
    IMG_SIZE_EFFNET = 380

    # Backbone 2: Multi-Stage Swin Transformer (Small)
    MODEL_SWIN_NAME = "swin_small_patch4_window7_224"
    IMG_SIZE_SWIN = 224

    # Training
    N_FOLDS = 5
    EPOCHS = (
        20  # Increased epochs to allow convergence (Cite solution_lesson_node_00008)
    )
    BATCH_SIZE = 32  # Increased batch size for stability

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # Relaxed patience (Cite solution_lesson_node_00006)

    # Inference
    TTA_FLIPS = ["horizontal", "vertical"]  # Exclude transpose as per strategy

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # =========================================================================
    # Utilities
    # =========================================================================
    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
