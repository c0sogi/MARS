import os
import torch


class Config:
    """
    Configuration class for the Cactus Identification Task.
    Defines hyperparameters, file paths, and system settings.
    """

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing available vCPUs (12 available)
    NUM_WORKERS = 8

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    # Storing artifacts in idea_3 folder
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Output File Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = 32
    NUM_CLASSES = 1  # Binary Classification

    # Normalization Constants (RGB)
    # Derived from Data Analysis:
    # Mean: R=128.37, G=115.25, B=119.40 -> [0.5034, 0.4520, 0.4683]
    # Std:  R=38.60,  G=35.68,  B=39.15  -> [0.1514, 0.1399, 0.1535]
    MEAN = [0.5034, 0.4520, 0.4683]
    STD = [0.1514, 0.1399, 0.1535]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Shallow Steerable Equivariant ResNet
    MODEL_NAME = "SteerableEquivariantResNet"
    # Base channel width for the equivariant layers
    HIDDEN_DIM = 48

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Extended training duration for Mixup convergence
    EPOCHS = 30
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    OPTIMIZER = "AdamW"

    # Scheduler (Cosine Annealing)
    SCHEDULER_MIN_LR = 1e-6

    # =========================================================================
    # Regularization & Augmentation
    # =========================================================================
    # Mild Mixup Regularization
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Inference Strategy
    # 4-view Test Time Augmentation (Original, HFlip, VFlip, Rot180)
    USE_TTA = True

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500

    @staticmethod
    def setup_directories():
        """
        Ensures that the necessary working and submission directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup_directories()
