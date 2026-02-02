import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Soft-Gated Multi-Task ResNet18 U-Net with Global Batch Optimization.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission File Path
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256

    # Input Channels:
    # 3 Channels for Ash False Color Composite (t=4)
    # 3 Channels for Temporal Difference (t=4 - t=3)
    IN_CHANNELS = 6

    # Bands required to construct Ash Composite (T11, T13, T14, T15)
    REQUIRED_BANDS = [11, 13, 14, 15]

    # ==========================================
    # Model Architecture
    # ==========================================
    ENCODER_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Compute: A100 40GB allows for larger batch sizes
    BATCH_SIZE = 64

    # Workers: 12 vCPUs available
    NUM_WORKERS = 12

    # Training Duration: Minimum 20 epochs as per Idea 4
    EPOCHS = 20

    # Optimization
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Post-Processing
    THRESHOLD = 0.5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=42):
        """
        Sets random seeds for reproducibility across Python, Numpy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize environment with fixed seed
Config.set_seed(Config.SEED)
