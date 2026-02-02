import os
import torch


class Config:
    """
    Configuration class for the Diabetic Retinopathy Detection Task.
    Centralizes all file paths, hyperparameters, and global settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SIZE = 100  # Number of samples to use in debug mode

    # ==========================================
    # Compute Settings
    # ==========================================
    # Use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of data loading workers (12 vCPUs available)
    NUM_WORKERS = 4

    # ==========================================
    # File Paths
    # ==========================================
    # Root input directory
    INPUT_DIR = "./input"

    # Metadata directory (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = "./working"
    # Cache directory for this specific idea/experiment
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Specific output file paths
    MODEL_PATH = os.path.join(CACHE_DIR, "resnet18_regression.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model & Data Hyperparameters
    # ==========================================
    # ResNet18 standard input size
    IMAGE_SIZE = 224

    # Regression output (1 neuron)
    NUM_OUTPUTS = 1

    # Training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Early stopping configuration
    PATIENCE = 4

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
