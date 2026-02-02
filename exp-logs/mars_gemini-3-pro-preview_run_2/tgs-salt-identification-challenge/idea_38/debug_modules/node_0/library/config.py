import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Salt Segmentation Task.
    Implements the FP32-Stabilized Marginalized-Distillation strategy settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea to ensure isolation
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_38")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    TEACHER_CHECKPOINT_DIR = os.path.join(CACHE_DIR, "teacher_checkpoints")
    STUDENT_CHECKPOINT_DIR = os.path.join(CACHE_DIR, "student_checkpoints")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    ORIG_SIZE = 101
    # Pad to 128x128 to be divisible by 32 (standard for U-Net/ResNet architectures)
    PAD_SIZE = 128
    CHANNELS = 1  # Grayscale input

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    FOLDS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # Training Horizons
    EPOCHS_TEACHER = 50
    EPOCHS_STUDENT = 50

    # Optimization
    WEIGHT_DECAY = 1e-2
    EARLY_STOPPING_PATIENCE = 10

    # =========================================================================
    # Strategy Specifics (Marginalized Distillation)
    # =========================================================================
    # Depths (in std devs) to scan during marginalization step
    MARGINALIZATION_DEPTHS = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # Augmentation Parameters (Elastic Transform is critical)
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6
    AUG_PROB = 0.5

    # =========================================================================
    # Compute & Debugging
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to a small integer (e.g., 100) to debug pipeline on a subset of data
    # Set to None for full training
    MAX_SAMPLES = None
    DEBUG = False

    @staticmethod
    def setup():
        """
        Initializes the environment: creates directories and sets random seeds.
        """
        # Create necessary directories
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.TEACHER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.STUDENT_CHECKPOINT_DIR, exist_ok=True)

        # Set reproducible seeds
        Config.set_seed(Config.SEED)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets seeds for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
