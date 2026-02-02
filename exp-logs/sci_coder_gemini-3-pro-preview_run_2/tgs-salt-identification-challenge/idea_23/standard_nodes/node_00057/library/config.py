import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the Salt Segmentation Task using ResNet34-WideLinkNet
    and Ensemble-Distilled Noisy Student training.
    """

    # ==========================================
    # Global Constants
    # ==========================================
    SEED = 42
    PROJECT_NAME = "idea_23"

    # ==========================================
    # File Paths
    # ==========================================
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Writable working directory
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOGS_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission output path (as per requirements)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Input Data Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV_PATH = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Original image dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Model input dimensions (padded for stride 32 compatibility)
    IMG_SIZE = 128
    CHANNELS = 1  # Grayscale input

    # Data Loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    BACKBONE = "resnet34"
    PRETRAINED = True
    DEPTH_EMBED_DIM = 32

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stage 1: Teacher Ensemble Training
    EPOCHS_STAGE1 = 50
    TEACHER_FOLDS = 5

    # Bernoulli Depth Masking Probability (for Teacher robustness)
    BERNOULLI_DEPTH_PROB = 0.5

    # Stage 3: Student Distillation Training
    EPOCHS_STAGE3 = 50

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the working environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
