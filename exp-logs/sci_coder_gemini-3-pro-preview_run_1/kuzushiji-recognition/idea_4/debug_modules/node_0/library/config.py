import os
import torch
import numpy as np
import random


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea/experiment
    WORKING_DIR = "./working/idea_4"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    # --- Model Hyperparameters ---
    BACKBONE = "convnext_base"
    # 1024x1024 input resolution as requested
    IMG_SIZE = 1024
    # Number of classes will be determined dynamically, but setting a default based on unicode map size
    # The unicode_translation.csv has 4782 lines, but training data might have fewer.
    # We usually map to the full set or just the training set.
    # Let's set a placeholder that the training script can update or use.
    NUM_CLASSES = 4782

    # --- Training Settings ---
    EPOCHS = 40
    # Batch size of 4 is conservative for ConvNeXt-Base @ 1024x1024 on 40GB VRAM
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4

    # --- Inference Settings ---
    CONF_THRESHOLD = 0.1
    MAX_DETECTIONS = 1200

    # --- Debugging/Development ---
    # Set to True to use a small subset of data for quick pipeline testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup_reproducibility(cls):
        """
        Sets the random seeds for python, numpy, and torch to ensure reproducibility.
        """
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed(cls.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def create_dirs(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize environment immediately upon import
Config.setup_reproducibility()
Config.create_dirs()
