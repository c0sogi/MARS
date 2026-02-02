import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/optimized"
    SUBMISSION_DIR = "./submission"

    # Input Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Sample Submission (for reference/IDs)
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    IMAGE_SIZE = 32
    CHANNELS = 3
    NUM_CLASSES = 1

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick pipeline testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    WEIGHT_DECAY = 1e-4

    # Cosine Annealing Scheduler
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Ensemble Strategy
    # ==========================================
    # We use a heterogeneous ensemble of ResNet and DenseNet
    ARCHITECTURES = ["resnet", "densenet"]
    # Seeds for averaging to reduce stochastic noise
    SEEDS = [0, 1, 2, 3, 4]

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of workers for DataLoader
    NUM_WORKERS = 4

    @staticmethod
    def setup_directories():
        """
        Creates the necessary working and submission directories if they do not exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def get_model_path(architecture, seed):
        """
        Generates the file path for saving/loading a model checkpoint.

        Args:
            architecture (str): The name of the architecture (e.g., 'resnet').
            seed (int): The random seed used for this model instance.

        Returns:
            str: Full path to the .pth file.
        """
        filename = f"{architecture}_seed_{seed}.pth"
        return os.path.join(Config.WORKING_DIR, filename)


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
