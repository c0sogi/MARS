import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the SRM-EfficientNet Steganalysis task.
    """

    # --- General Configuration ---
    PROJECT_NAME = "Steganalysis_SRM_EfficientNet"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    COVER_DIR = os.path.join(INPUT_DIR, "Cover")
    JMIPOD_DIR = os.path.join(INPUT_DIR, "JMiPOD")
    JUNIWARD_DIR = os.path.join(INPUT_DIR, "JUNIWARD")
    UERD_DIR = os.path.join(INPUT_DIR, "UERD")
    TEST_DIR = os.path.join(INPUT_DIR, "Test")

    # Output Directories
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # File Paths for Outputs
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Architecture ---
    BACKBONE = "efficientnet_b0"
    NUM_CLASSES = 1
    # SRM Filter Bank settings
    SRM_IN_CHANNELS = 1  # Input is grayscale (Y channel)
    SRM_OUT_CHANNELS = 30  # 30 filters from SRM
    USE_GEM_POOLING = True

    # --- Data Preprocessing ---
    IMAGE_SIZE = 512
    USE_Y_CHANNEL = True  # Extract Luminance channel only

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    EARLY_STOPPING_PATIENCE = 3

    # Class Balancing (Weighted Sampling)
    # Target ratio of Cover samples in a batch.
    # 0.5 means 50% Cover, 50% Stego (effectively oversampling Cover by 3x)
    COVER_BATCH_RATIO = 0.5

    # --- Metric (Weighted AUC) ---
    TPR_THRESHOLDS = [0.0, 0.4, 1.0]
    TPR_WEIGHTS = [2, 1]

    @classmethod
    def setup(cls):
        """
        Performs initial setup: creates directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        cls.set_seed(cls.SEED)

        print(f"Configuration setup complete. Device: {cls.DEVICE}")
        print(f"Working Directory: {cls.WORKING_DIR}")

    @staticmethod
    def set_seed(seed):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def update(cls, **kwargs):
        """
        Updates configuration parameters dynamically.
        Useful for hyperparameter tuning or command-line overrides.
        """
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
                print(f"Config updated: {k} = {v}")
            else:
                print(f"Warning: Config has no attribute '{k}'")
