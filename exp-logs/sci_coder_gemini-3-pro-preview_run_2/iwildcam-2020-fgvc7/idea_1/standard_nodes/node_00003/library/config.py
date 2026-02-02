import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    MEGADETECTOR_JSON = os.path.join(
        INPUT_DIR, "iwildcam2020_megadetector_results.json"
    )
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data & Model Hyperparameters
    # --------------------------------------------------------------------------
    # Image processing
    IMG_SIZE = (448, 448)
    CHANNELS = 3

    # Classification
    # Category IDs range from 0 to 675.
    # 0 is empty, 1-675 are species. Total 676 classes.
    NUM_CLASSES = 676

    # Training
    BACKBONE = "resnet50"  # Using ResNet50 as the fixed feature extractor
    BATCH_SIZE = 32  # Decreased for larger images
    LEARNING_RATE = 1e-4  # Lower for fine-tuning
    EPOCHS = 5  # Short training since backbone is frozen

    # --------------------------------------------------------------------------
    # System Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 8  # Utilize available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def make_dirs():
        """
        Ensures necessary working and submission directories exist.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
