import os
import torch


class Config:
    """
    Configuration class for the PCA-Enhanced Quantile-Elastic Pipeline.
    Centralizes all constants, paths, and hyperparameters.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMAGE_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGE_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Image Processing & Feature Extraction
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 224  # Input resolution for EfficientNet-B0
    NUM_SLICES = 5  # Number of slices to select based on variance
    BATCH_SIZE = 32  # Batch size for feature extraction
    NUM_WORKERS = 2  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Dimensionality Reduction
    # ==========================================
    PCA_COMPONENTS = 30  # Target dimensions for PCA compression

    # ==========================================
    # Solver Hyperparameters
    # ==========================================
    # 1. FVC Predictor (Linear Quantile Regressor)
    QUANTILE = 0.5  # Median regression
    QR_MAX_ITER = 2000  # Max iterations for the solver

    # 2. Uncertainty Predictor (Elastic Net)
    EN_ALPHA = 0.5  # Regularization strength
    EN_L1_RATIO = 0.5  # Mix between L1 and L2 (0.5 = equal mix)
    EN_MAX_ITER = 2000  # Max iterations for Elastic Net

    # ==========================================
    # Metric Constants
    # ==========================================
    METRIC_CLIP_SIGMA = 70
    METRIC_MAX_DELTA = 1000

    # ==========================================
    # Caching Logic
    # ==========================================
    LOAD_CACHED_DATA = (
        True  # If True, attempts to load .npy/.parquet files from WORKING_DIR
    )

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
