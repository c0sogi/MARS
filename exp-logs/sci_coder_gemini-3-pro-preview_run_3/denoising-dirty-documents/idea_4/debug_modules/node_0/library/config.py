import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the DnCNN denoising pipeline.
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dncnn_model.pth")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_patches.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_patches.npy")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    PATCH_SIZE = 50
    STRIDE = 20  # Stride for extracting patches (controls overlap)
    AUGMENT_DATA = True  # Whether to apply flips/rotations

    # Debugging / Flexibility
    # Set to an integer (e.g., 100) to limit dataset size for faster debugging.
    # Set to None to use the full dataset.
    MAX_TRAIN_IMAGES = None
    MAX_VAL_IMAGES = None

    # -------------------------------------------------------------------------
    # Model Hyperparameters (DnCNN)
    # -------------------------------------------------------------------------
    DEPTH = 17  # Number of convolutional layers
    N_CHANNELS = 64  # Number of feature maps in hidden layers
    IN_CHANNELS = 1  # Input channels (Grayscale)
    KERNEL_SIZE = 3
    PADDING = 1  # Keeps spatial dimensions constant

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-8
    EARLY_STOPPING_PATIENCE = 5

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def initialize(cls):
        """
        Sets up the environment: creates necessary directories and sets random seeds.
        """
        # Ensure working and submission directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize the configuration environment upon import
Config.initialize()
