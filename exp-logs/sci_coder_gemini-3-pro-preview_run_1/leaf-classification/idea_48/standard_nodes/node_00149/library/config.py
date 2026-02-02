import os
import numpy as np


class Config:
    """
    Global configuration for the Leaf Identification Task.
    Implements the 'Polarity-Corrected Non-Linear High-Precision OAS Discriminant' strategy.
    """

    # ==========================================
    # Global Constants & Reproducibility
    # ==========================================
    SEED = 42
    # Strict enforcement of Double Precision as per Lesson 00057
    DTYPE = np.float64

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working Directory for Caching (Optimized)
    WORKING_DIR = "./working/idea_opt"

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Image Processing Configuration
    # ==========================================
    # Polarity Correction: Invert images to ensure leaf is foreground (white)
    # and background is black. Addresses Lesson 00145.
    INVERT_IMAGES = True

    # ==========================================
    # Feature Extraction Configuration
    # ==========================================
    # List of geometric features to be explicitly computed and appended to tabular data
    # Optimized for Parsimony (Cite Lesson 00140) and Boundedness (Cite Lesson 00142)
    # Includes Macro-Geometric Triad (Cite Lesson 00120) and Absolute Size (Cite Lesson 00118)
    GEOMETRIC_FEATURES = [
        "equivalent_diameter",
        "aspect_ratio",
        "extent",
        "solidity",
        "roundness",
        "eccentricity",
    ]

    # ==========================================
    # Model & Training Configuration
    # ==========================================
    # Model Type: Oracle Approximating Shrinkage
    MODEL_TYPE = "OAS"

    # OAS Hyperparameters
    # We manually center data to ensure geometric consistency
    ASSUME_CENTERED = True

    # Debugging / Runtime Control
    DEBUG = False
    # If DEBUG is True, limit dataset size to this number
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup(cls):
        """
        Ensure necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, name):
        """
        Generate a path for a cached file in the working directory.

        Args:
            name (str): Name of the file (e.g., 'X_train.parquet')

        Returns:
            str: Full path to the cached file.
        """
        return os.path.join(cls.WORKING_DIR, name)
