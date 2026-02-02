import os
import numpy as np


class Config:
    """
    Configuration class for the Metric-Augmented Dynamic Generative Ensemble (MADGE).
    Defines file paths, random seeds, and model hyperparameters.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_SEED = 42

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories
    CACHE_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Schema
    # =========================================================================
    ID_COL = "id"
    TARGET_COL = "species"
    IMAGE_PATH_COL = "image_path"

    # Feature definitions
    # Note: The dataset has 3 groups of 64 features each (Margin, Shape, Texture)
    N_FEATURES_PER_GROUP = 64
    TOTAL_FEATURES = 192

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # 1. Neighborhood Components Analysis (NCA)
    # Projects features into a discriminative space before LDA
    NCA_COMPONENTS = 99  # Set to number of classes
    NCA_INIT = "auto"
    NCA_MAX_ITER = 500
    NCA_TOL = 1e-5

    # 2. Linear Discriminant Analysis (LDA)
    # Used as the generative expert (Global and NCA-transformed)
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # 3. Logistic Regression
    # Used as the discriminative expert
    LOGREG_SOLVER = "lbfgs"
    LOGREG_MAX_ITER = 2000
    LOGREG_JOBS = -1
    # Dense logarithmic grid for C optimization (10 values logarithmically spaced)
    LOGREG_C_GRID = np.logspace(-4, 4, 20).tolist()

    @classmethod
    def ensure_directories(cls):
        """
        Creates the necessary working directories for cache and submission
        if they do not already exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize environment by ensuring directories exist
Config.ensure_directories()
