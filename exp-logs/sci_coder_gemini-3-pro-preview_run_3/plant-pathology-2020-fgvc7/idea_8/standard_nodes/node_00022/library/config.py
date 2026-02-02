import os
import torch
import numpy as np
import random


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directories & File Paths
    # ====================================================
    # Input directories
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata directories (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    # We use idea_8 as the working directory for this specific iteration
    OUTPUT_DIR = "./working/idea_8"
    SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ====================================================
    # Data Information
    # ====================================================
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # ====================================================
    # Model & Training Hyperparameters
    # ====================================================
    # EfficientNet-B4 settings
    MODEL_EFFNET = "tf_efficientnet_b4_ns"
    IMG_SIZE_EFFNET = 380

    # MaxViT-Tiny settings
    MODEL_MAXVIT = "maxvit_tiny_tf_224.in1k"
    IMG_SIZE_MAXVIT = 224

    # Training settings
    BATCH_SIZE = 16  # Adjusted for A100 memory with these models
    EPOCHS = 25
    PATIENCE = 10  # Relaxed patience as per strategy
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 1000

    # TTA Settings
    TTA_FLIP_HORIZONTAL = True
    TTA_FLIP_VERTICAL = False  # Explicitly excluded per strategy
    TTA_TRANSPOSE = False  # Explicitly excluded per strategy

    @staticmethod
    def seed_everything(seed: int = 42):
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
