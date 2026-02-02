import os
import numpy as np


class Config:
    """
    Configuration module for the Integral-Inertial High-Precision OAS Discriminant.
    Defines global constants, file paths, feature groups, and hyperparameters.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    FLOAT_PRECISION = np.float64

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory for deterministic data processing (Idea 49)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_49")

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Image Processing Configuration
    # ==========================================
    # Invert images to ensure leaves are foreground (white) on background (black)
    # This is crucial for correct moment calculation.
    INVERT_IMAGES = True

    # ==========================================
    # Feature Definitions
    # ==========================================

    # 1. Raw Tabular Features (Provided in dataset)
    # 64 attributes per feature type
    MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
    SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
    TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

    RAW_TABULAR_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

    # 2. Integral-Inertial Features (To be extracted)
    # Derived from Image Moments (Inertia Tensor) and Topological Primitives.
    # We avoid boundary-fitting features like fitEllipse in favor of moment-based axes.
    EXTRACTED_FEATURES = [
        "Area",  # M00
        "Inertial_Major_Axis",  # Derived from eigenvalues of central moments
        "Inertial_Minor_Axis",  # Derived from eigenvalues of central moments
        "Eccentricity",  # Ratio of eigenvalues
        "AABB_Aspect_Ratio",  # Scanner frame orientation signal
        "AABB_Extent",  # Area / BoundingRectArea
        "Perimeter",  # Contour length
        "Convex_Perimeter",  # Convex Hull length
        "Solidity",  # Area / ConvexArea
        "Convexity",  # ConvexPerimeter / Perimeter
    ]

    # 3. Excluded Features
    # Features explicitly rejected due to noise sensitivity, redundancy, or instability.
    EXCLUDED_FEATURES = [
        "Equivalent_Diameter",  # Redundant with Area (monotonic transformation)
        "Roundness",  # Redundant with Area/Perimeter
        "Compactness",  # Often noisy for complex leaf margins
        "Hu_Moments",  # Highly sensitive to pixelation noise
        "Hu_Moment_1",
        "Hu_Moment_2",
        "Hu_Moment_3",
        "Hu_Moment_4",
        "Hu_Moment_5",
        "Hu_Moment_6",
        "Hu_Moment_7",
        "Ellipse_Angle",  # Unstable output from cv2.fitEllipse
        "Ellipse_Center_X",
        "Ellipse_Center_Y",
    ]

    # ==========================================
    # Model & Pipeline Hyperparameters
    # ==========================================
    # OAS Estimator settings
    # We assume centered data because we manually calculate residuals (X - mean).
    OAS_ASSUME_CENTERED = True

    # Preprocessing flags
    APPLY_POWER_TRANSFORM = True  # Yeo-Johnson to Gaussianize features
    APPLY_SCALING = True  # StandardScaler to normalize variance
