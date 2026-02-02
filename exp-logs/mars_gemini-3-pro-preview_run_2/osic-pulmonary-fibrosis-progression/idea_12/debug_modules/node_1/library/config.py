import os


class Config:
    """
    Configuration for the MIP-Enhanced Zonal-Quantile Pipeline.
    Centralizes all constants, paths, and hyperparameters.
    """

    # ==========================
    # General Configuration
    # ==========================
    SEED = 42
    N_JOBS = 12  # Number of vCPUs available

    # Debugging / Development
    # Set DEBUG = True to run on a small subset of patients for testing
    DEBUG = False
    DEBUG_SIZE = 20

    # ==========================
    # Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories
    # CACHE_DIR: Stores processed features (npy/parquet) to avoid re-computation
    CACHE_DIR = "./working/idea_12"
    # SUBMISSION_DIR: Stores the final submission file
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Image Processing & Feature Extraction
    # ==========================
    # Input Image Specs
    IMAGE_SIZE = 224

    # Lung Masking Thresholds (Hounsfield Units)
    HU_MIN = -1000
    HU_MAX = -400

    # Volumetric Priors
    HISTOGRAM_BINS = 4  # Number of bins for Global Density Histogram

    # Backbone for Visual Features
    BACKBONE_NAME = "efficientnet_b0"

    # ==========================
    # Dimensionality Reduction
    # ==========================
    # Target number of components for PCA
    # Compressing concatenated features (3 Axial + 1 MIP + Hist + Clinical)
    N_COMPONENTS = 40

    # ==========================
    # Model Hyperparameters
    # ==========================
    # 1. FVC Predictor: Linear Quantile Regressor (Target: Median, q=0.5)
    # Alpha: Regularization strength (L1 penalty on coefficients)
    QREG_ALPHA = 0.5
    # Solver: 'highs' is recommended for linear programming in scipy/sklearn
    QREG_SOLVER = "highs"

    # 2. Uncertainty Predictor: Elastic Net Regressor
    # Target: Absolute Residuals |y_true - y_pred|
    # Alpha: Constant that multiplies the penalty terms
    ENET_ALPHA = 0.1
    # L1_Ratio: The mixing parameter, with 0 <= l1_ratio <= 1.
    # For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.
    ENET_L1_RATIO = 0.5

    # Optimization
    MAX_ITER = 10000  # Max iterations for sklearn solvers

    # ==========================
    # Post-Processing & Metric
    # ==========================
    # Metric Constants
    MIN_CONFIDENCE = 70.0  # Clipped sigma
    MAX_ERROR_METRIC = 1000.0  # Delta clipping

    @staticmethod
    def mkdirs():
        """Creates necessary working directories."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
