import os
import torch
import random
import numpy as np


class Config:
    # =========================
    # Path Configuration
    # =========================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Specific cache directory for this idea iteration
    CACHE_DIR = "./working/idea_10"

    # Output paths
    WORKING_DIR = "./working"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================
    # Data Preprocessing
    # =========================
    # Standard Bone Window
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Input Dimensions
    IMAGE_SIZE = (256, 256)
    SEQ_LENGTH = 64  # Number of slices sampled per exam
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # =========================
    # Model Configuration
    # =========================
    BACKBONE = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 7  # C1-C7
    EMBEDDING_DIM = 512  # Feature dimension from ResNet18

    # =========================
    # Training Configuration
    # =========================
    SEED = 42
    BATCH_SIZE = 8
    NUM_EPOCHS = 10

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler: Decoupled Cosine Annealing
    # T_max will be calculated as T_MAX_MULTIPLIER * NUM_EPOCHS
    T_MAX_MULTIPLIER = 1.5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.WORKING_DIR, exist_ok=True)
