import os
import random
import numpy as np
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data & Preprocessing
    # -------------------------------------------------------------------------
    ORIGINAL_SIZE = 96
    CROP_SIZE = 48  # Center crop size (Hard Attention)
    IMG_SIZE = 48  # Input size to the model

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Heterogeneous Ensemble members
    MODEL_NAMES = ["densenet121", "resnet50"]

    # Architectural modification: Replace 7x7 stride-2 stem with 3x3 stride-1
    MODIFY_STEM = True

    PRETRAINED = True
    NUM_CLASSES = 1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256  # A100 40GB can handle large batches of 48x48 images
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 6  # Relaxed patience for early stopping

    # -------------------------------------------------------------------------
    # Compute & System
    # -------------------------------------------------------------------------
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        """
        Creates necessary directories and sets fixed seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Enforce deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
