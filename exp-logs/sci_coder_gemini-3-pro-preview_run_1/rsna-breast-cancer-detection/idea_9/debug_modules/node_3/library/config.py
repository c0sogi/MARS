import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    PROJECT_NAME = "Breast_Cancer_Detection_Siamese"
    IDEA_NAME = "idea_9"
    SEED = 42
    DEBUG = False  # Set to True for fast debugging with small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # =========================================================================
    # File Paths
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    IMG_SIZE = (768, 768)  # Height, Width
    NUM_CHANNELS = 3  # Image + Age + Implant

    # Normalization Constants (Approximate for Mammography)
    # Note: We use instance-level normalization or standard scaling in the dataset class,
    # but these can be used for standardization if needed.
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "tf_efficientnet_b2"
    PRETRAINED = True
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2

    # Siamese Specifics
    USE_SIAMESE = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Conservative for 768x768 on A100
    NUM_WORKERS = 4  # 12 vCPUs available
    # Cite debug_lesson_9: Disable pin_memory to prevent AcceleratorError/OOM
    PIN_MEMORY = False

    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Loss Function
    # Pos weight of 47.0 to handle ~1:47 imbalance
    POS_WEIGHT = 47.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Optimization
    MAX_GRAD_NORM = None  # Disable gradient clipping as per strategy
    ACCUMULATE_GRAD_BATCHES = 1

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MODE = "max"  # Monitor pF1 score

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed):
        """
        Sets seeds for reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Execute setup when module is imported to ensure directories exist
Config.setup()
