import os
import torch
import random
import numpy as np


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Hyperparameters ---
    BACKBONE = "resnet50"
    IMG_SIZE = 1024  # Square resolution as per idea
    # Number of classes based on EDA (Unique Character Classes in Train)
    # We will likely map these dynamically, but defining the upper bound from unicode map is safer
    # Unicode translation file has 4782 lines.
    # However, EDA shows 3848 unique classes in training.
    # We will use 4782 to cover the full potential vocabulary or map strictly to training.
    # Let's set a default here, usually determined by the LabelEncoder length.
    NUM_CLASSES = 4782

    # Head Channels
    HM_CHANNELS = 1  # Objectness Heatmap
    REG_CHANNELS = 4  # Regression: 2 for offset, 2 for dimensions (w, h)

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 4  # Adjusted for 1024x1024 on A100 to avoid OOM
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler
    WARMUP_EPOCHS = 3

    # --- Inference Hyperparameters ---
    CONF_THRESHOLD = 0.1
    TOP_K = 1200  # Max predictions per page as per task description

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed immediately upon import
seed_everything(Config.SEED)
