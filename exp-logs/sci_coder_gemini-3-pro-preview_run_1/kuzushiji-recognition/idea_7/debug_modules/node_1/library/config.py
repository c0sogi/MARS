import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_7"

    # Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

    # Image Directories
    TRAIN_IMGS_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMGS_DIR = os.path.join(INPUT_DIR, "test_images")

    # Cache Files
    LABEL_ENCODER_PATH = os.path.join(WORK_DIR, "label_encoder.npy")
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==============================
    # Data Configuration
    # ==============================
    IMG_SIZE = 1024  # Resizing target (Square)
    IN_CHANNELS = 3

    # Normalization (ImageNet Standards)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==============================
    # Model Configuration
    # ==============================
    MODEL_NAME = "convnext_tiny"
    # Number of classes based on Training EDA (3848 unique chars)
    # This serves as a default; the pipeline may verify this against the label encoder.
    NUM_CLASSES = 3848

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 8  # Adjusted for 1024x1024 on A100 GPU
    NUM_EPOCHS = 35  # Extended training for high cardinality convergence
    LEARNING_RATE = 2e-4  # Initial LR for AdamW
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    NUM_WORKERS = 4  # Data loading workers

    # Scheduler settings (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    MIN_LR = 1e-6

    # ==============================
    # Inference Hyperparameters
    # ==============================
    CONF_THRESHOLD = 0.1  # Minimum confidence to propose a character
    MAX_DETECTIONS = 1200  # Submission limit per page

    # ==============================
    # Hardware & Debugging
    # ==============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = False  # Set True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of images to use in debug mode


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
