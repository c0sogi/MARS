import os
import torch
import random
import numpy as np


class Config:
    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # --- Compute ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Artifacts and Outputs
    WORKING_DIR = "./working/idea_2"
    OUTPUT_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(WORKING_DIR, "convnext_base_best.pth")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # --- Model ---
    # Using ConvNeXt Base as per the strategy
    MODEL_NAME = "convnext_base"
    NUM_CLASSES = 3474

    # --- Data ---
    # Increased resolution for artwork details
    IMAGE_SIZE = 384

    # --- Training ---
    EPOCHS = 10
    # A100 40GB can handle larger batches even with 384x384
    BATCH_SIZE = 64

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Loss
    # Label smoothing to handle noisy annotations
    LABEL_SMOOTHING = 0.05

    # Inference
    # Threshold for multi-label classification (will be tuned on val set)
    DEFAULT_THRESHOLD = 0.3

    @staticmethod
    def seed_everything(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # Set to True if input sizes are constant for speed
        )


# Initialize directories and seed on import
Config.seed_everything(Config.SEED)
