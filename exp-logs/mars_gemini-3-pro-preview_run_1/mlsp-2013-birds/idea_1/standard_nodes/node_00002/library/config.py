import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Bird Species Prediction task.
    Defines paths, hyperparameters, and utility methods.
    """

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Generated Metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Source Data
    # 100-dimensional histogram features
    HISTOGRAM_FILE_PATH = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    INPUT_DIM = 100  # Feature vector size from histogram_of_segments.txt
    NUM_CLASSES = 19  # Number of bird species
    ID_MULTIPLIER = 100  # Used for submission ID formatting: rec_id * 100 + species_id

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # MLP Architecture
    HIDDEN_LAYERS = [64, 32]
    DROPOUT_RATE = 0.5

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # ==========================================
    # Debugging / Development
    # ==========================================
    # If set to an integer, limits the number of samples for quick testing
    DEBUG_SUBSET_SIZE = None

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_device(cls):
        """
        Returns the appropriate PyTorch device (CUDA or CPU).
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_directories(cls):
        """
        Ensures that working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
