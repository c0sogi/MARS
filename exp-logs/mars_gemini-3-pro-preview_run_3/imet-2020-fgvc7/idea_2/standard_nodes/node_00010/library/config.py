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
    WORKING_DIR = "./working/idea_3"
    OUTPUT_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(WORKING_DIR, "resnet50d_best.pth")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # --- Model ---
    # Using ResNet50d as per Lesson 00005 (Capacity)
    MODEL_NAME = "resnet50d"
    NUM_CLASSES = 3474

    # --- Data ---
    # Standard resolution to allow full dataset training (Lesson 00008)
    IMAGE_SIZE = 224

    # --- Training ---
    EPOCHS = 10
    # A100 40GB can handle larger batches with 224x224
    BATCH_SIZE = 128

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Loss
    # Disable label smoothing for sparse targets (Lesson 00009)
    LABEL_SMOOTHING = 0.0

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
