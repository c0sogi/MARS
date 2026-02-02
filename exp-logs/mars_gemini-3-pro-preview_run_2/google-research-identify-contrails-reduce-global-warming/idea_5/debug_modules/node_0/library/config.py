import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Contrail Identification pipeline.
    Defines file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 5 (Context-Enhanced ResNet18 U-Net)
    WORKING_DIR = "./working/idea_5"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    N_CHANNELS = 6  # 3 Ash Channels + 3 Temporal Difference Channels

    # Temporal sequence details
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # ==========================================
    # Model Configuration
    # ==========================================
    ENCODER_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32  # Optimized for A100 40GB with ResNet18 U-Net
    NUM_WORKERS = 4  # 12 vCPUs available

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Loss Weights (if applicable)
    BCE_WEIGHT = 1.0
    DICE_WEIGHT = 1.0

    # ==========================================
    # Inference & Post-processing
    # ==========================================
    THRESHOLD = 0.5

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
