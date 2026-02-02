import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    # Root directory for read-only input data
    INPUT_ROOT = "./input"

    # Directory containing the generated metadata CSVs (train.csv, val.csv, test.csv)
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 24)
    # Used for caching processed data, checkpoints, and logs
    WORKING_DIR = "./working/idea_24"

    # Sub-directories for organization
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"  # Standard location for final submission

    # DICOM directories (relative to INPUT_ROOT)
    TRAIN_DICOM_DIR = os.path.join(INPUT_ROOT, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_ROOT, "test")

    # ==========================================
    # 2. Data Configuration
    # ==========================================
    # Image resolution - strictly 224x224 to avoid upscaling artifacts
    IMG_SIZE = 224

    # Number of channels per view (Tri-Slab RGB)
    IN_CHANNELS = 3

    # Tabular features to use
    TABULAR_COLS = ["Weeks", "Percent", "Age", "Sex", "SmokingStatus"]

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Backbone network
    BACKBONE_NAME = "efficientnet_b0"

    # Dimensionality of the backbone output (EfficientNet-B0 GAP output)
    # We maintain this native dimensionality for the tabular projection
    FEATURE_DIM = 1280

    # Dropout rate for the regression head
    DROPOUT_RATE = 0.2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size (A100 has 40GB, can handle larger batches, but 32 is stable for convergence)
    BATCH_SIZE = 32

    # Number of workers for data loading
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 50

    # Early Stopping
    # Strict patience as per idea description to prevent overfitting on small cohort
    PATIENCE = 8

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # 5. Metric & Loss Constants
    # ==========================================
    # Constants for Modified Laplace Log Likelihood
    MIN_SIGMA = 70.0  # Clipped confidence (ml)
    MAX_DELTA = 1000.0  # Threshold for absolute error (ml)

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize directories immediately upon import
Config.setup()
