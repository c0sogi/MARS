import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directories
    # Note: idea_14 is the designated folder for this experiment
    CACHE_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    # Image Preprocessing
    IMG_SIZE = 224  # Input size for EfficientNet-B0
    SLICE_COUNT = 3  # Number of stratified slices (Top, Middle, Bottom)
    HU_MIN = -1000  # Minimum Hounsfield Unit (Lung window)
    HU_MAX = -400  # Maximum Hounsfield Unit (Lung window)

    # Morphological Profiling
    POLY_ORDER = 3  # Degree of polynomial for Area/Density Z-axis curves

    # Texture Analysis
    PCA_COMPONENTS = 30  # Number of components for texture dimensionality reduction

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # FVC Prediction (Quantile Regression)
    QUANTILE = 0.5  # Median regression for robust central tendency

    # Uncertainty Prediction (Elastic Net)
    ELASTIC_NET_ALPHA = 1.0  # Regularization strength
    ELASTIC_NET_L1_RATIO = 0.5  # Balance between L1 and L2 penalty

    # ==========================================
    # Runtime & Training Settings
    # ==========================================
    SEED = 42
    N_JOBS = 12  # Number of vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 20  # Number of patients to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_seed(cls, seed=None):
        """
        Sets random seeds for reproducibility across Python, Numpy, and Torch.

        Args:
            seed (int, optional): Specific seed to use. Defaults to Config.SEED.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cuDNN if needed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
