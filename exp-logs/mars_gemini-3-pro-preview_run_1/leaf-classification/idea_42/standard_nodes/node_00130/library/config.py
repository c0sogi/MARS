import os
import numpy as np


class Config:
    # ==========================================
    # Global Random Seed
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Types and Precision
    # ==========================================
    # Critical for avoiding the 1e-7 metric floor in log-loss
    FLOAT_TYPE = np.float64
    INT_TYPE = np.int64

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory as required by the task description
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_42")

    # Submission output directory
    SUBMISSION_DIR = "./submission"

    # Image directory (images are located in input/images)
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata File Paths
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Column Definitions
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "species"
    IMAGE_PATH_COL = "file_path"

    # Pre-extracted feature prefixes
    # Each has 64 attributes (e.g., margin_1 ... margin_64)
    TABULAR_FEATURE_PREFIXES = ["margin", "shape", "texture"]

    # New Morphological Features to be extracted from images
    # These capture Absolute Scale, Rotated Envelope, and Internal Morphology
    MORPHOLOGICAL_FEATURES = [
        # Absolute Scale
        "Area",
        "Perimeter",
        "Convex_Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Equivalent_Diameter",
        # Rotated Envelope (Minimum Area Rectangle)
        "Min_Box_Width",
        "Min_Box_Height",
        "Min_Box_Area",
        # Internal Morphology (Distance Transform)
        "Inscribed_Circle_Radius",
        # Relative Shape / Dimensionless
        "Solidity",
        "Extent",
        "Min_Box_Aspect_Ratio",
        "Convexity",
        "Roundness",
        "Eccentricity",
    ]

    # ==========================================
    # Model Hyperparameters (OAS-LDA)
    # ==========================================
    # OAS Estimator settings
    # We assume data is centered manually via residuals (X - mu_y)
    OAS_ASSUME_CENTERED = True

    # ==========================================
    # Preprocessing Configuration
    # ==========================================
    # Yeo-Johnson Transformation
    YEO_JOHNSON_STANDARDIZE = False  # We apply standard scaling separately

    # ==========================================
    # Compute Resources
    # ==========================================
    NUM_WORKERS = 12  # Based on available vCPUs

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately when module is imported to ensure directories exist
Config.setup()
