import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # System & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 224
    NUM_CLASSES = 32093
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Output Configuration
    # ==========================================
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet18_arcface_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet50"
    EMBEDDING_SIZE = 512  # Output dimension of the backbone before the head

    # ArcFace Hyperparameters
    ARCFACE_MARGIN = 0.50
    ARCFACE_SCALE = 30.0

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # A100 40GB can handle larger batches
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # Sampling
    # Used for Inverse Square Root Sampling
    SAMPLING_POWER = 0.5

    @staticmethod
    def setup():
        """
        Initializes the environment: creates directories and sets random seeds.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
