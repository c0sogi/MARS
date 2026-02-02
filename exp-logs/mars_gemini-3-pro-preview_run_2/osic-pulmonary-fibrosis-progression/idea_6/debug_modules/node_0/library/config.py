import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Dual-Moment GLM-Quantile Pipeline.
    Centralizes all hyperparameters, file paths, and global settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    IDEA_NAME = "idea_6"

    # Debugging controls
    # Set DEBUG to True to run the pipeline on a small subset of data for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Generated in previous steps)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # DICOM Image Directories
    TRAIN_DCM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DCM_DIR = os.path.join(INPUT_DIR, "test")

    # Working Directory for Caching Intermediate Results (Features, Models)
    # Requirement: Ensure this directory exists
    CACHE_DIR = os.path.join("./working", IDEA_NAME)

    # Output Directory for Final Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing & Visual Backbone
    # -------------------------------------------------------------------------
    # Slice Selection Strategy: Variance-Based
    # We select the top N slices with the highest pixel variance to capture heterogeneity.
    N_SLICES = 5

    # EfficientNet-B0 Input Parameters
    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # DataLoader Settings
    BATCH_SIZE = 16
    NUM_WORKERS = 2  # Adjusted for the available vCPUs

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    # Dimensionality Reduction
    # We compress the concatenated Mean+Std deep features to avoid the curse of dimensionality
    # in the subsequent linear models.
    PCA_COMPONENTS = 40

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------

    # 1. FVC Predictor: Linear Quantile Regressor
    # Objective: Minimize L1 Norm (Mean Absolute Error) at the Median (q=0.5)
    # Inputs: PCA Features + Clinical Data + Interaction Terms
    QR_PARAMS = {
        "quantile": 0.5,  # Median regression
        "alpha": 0.01,  # L1 regularization strength (Lasso-like) on coefficients
        "solver": "highs",  # High-performance linear programming solver
        "fit_intercept": True,
    }

    # 2. Uncertainty Predictor: Gamma GLM
    # Objective: Minimize Gamma Deviance (Negative Log Likelihood)
    # Target: Absolute Residuals (|y_true - y_pred|) + epsilon
    # Inputs: PCA Features + Clinical Data + Time Horizon
    GLM_PARAMS = {
        "family": "Gamma",  # Models positive, right-skewed error distributions
        "link": "Log",  # Ensures positive predictions for sigma
        "alpha": 0.1,  # Regularization strength
        "l1_ratio": 0.5,  # ElasticNet mixing (0.5 = balanced L1/L2)
        "max_iter": 1000,
        "tol": 1e-4,
    }

    # Post-Processing
    # Clipping for confidence values as per metric definition
    CONFIDENCE_CLIP_MIN = 70

    # -------------------------------------------------------------------------
    # Compute Configuration
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls._set_seed(cls.SEED)

    @staticmethod
    def _set_seed(seed):
        """Sets the seed for all random number generators."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize setup immediately upon import
Config.setup()
