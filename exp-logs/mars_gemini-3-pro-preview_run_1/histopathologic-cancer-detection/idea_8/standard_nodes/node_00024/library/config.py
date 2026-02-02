import os
import torch


class Config:
    """
    Configuration for the Tumor Detection Task.
    Implements the settings for a Heterogeneous Ensemble of DenseNets.
    """

    # ==========================================
    # Global Seeds & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data

    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    # Note: The strategy involves using the full training set (train + val metadata)
    # for 5-fold CV.
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 48  # Input size (ROI crop)
    NUM_CLASSES = 1

    # Dataset limits for debugging (None = use all data)
    MAX_TRAIN_SAMPLES = 1000 if DEBUG else None
    MAX_TEST_SAMPLES = 1000 if DEBUG else None

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous ensemble of DenseNets
    ARCHITECTURES = ["densenet121", "densenet169"]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    EPOCHS = 20
    BATCH_SIZE = 256  # Large batch size for A100 and small images
    LEARNING_RATE = 1e-4
    PATIENCE = 6  # Relaxed patience for convergence
    LABEL_SMOOTHING = 0.05
    WEIGHT_DECAY = 0.01  # For AdamW

    # ==========================================
    # Inference / TTA
    # ==========================================
    TTA_STEPS = 4  # Original, HFlip, VFlip, Rot90

    # ==========================================
    # Hardware Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # Initialization
    # ==========================================
    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately on import
Config.setup()
