import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration for the Two-Stage Geometric Clustering and Gradient Boosting Pipeline.
    """

    # --------------------------------------------------------------------------
    # PATHS & DIRECTORIES
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # GLOBAL SETTINGS
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42
    NUM_WORKERS = 12  # Adjusted to available vCPUs

    # --------------------------------------------------------------------------
    # GEOMETRIC PIPELINE HYPERPARAMETERS
    # --------------------------------------------------------------------------
    # Ground Plane Removal (RANSAC)
    RANSAC_DIST_THRESH = 0.3  # Distance threshold to consider a point as ground
    RANSAC_ITERATIONS = 100

    # Euclidean Clustering (DBSCAN)
    DBSCAN_EPS = 0.6  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
    DBSCAN_MIN_SAMPLES = 5  # The number of samples in a neighborhood for a point to be considered as a core point

    # Proposal Filtering
    MIN_CLUSTER_POINTS = 10  # Minimum points to consider a valid object candidate
    MAX_CLUSTER_POINTS = (
        50000  # Maximum points (avoids processing walls/large terrain chunks)
    )

    # Ground Truth Matching (IoU)
    IOU_POS_THRESH = 0.4  # Proposals with IoU >= 0.4 are considered Positive matches
    IOU_NEG_THRESH = (
        0.1  # Proposals with IoU < 0.1 are considered Negative (Background)
    )

    # --------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------------------------
    # List of geometric features to extract for each cluster
    FEATURES = [
        # Eigenvalue-based shape descriptors
        "eigen_1",
        "eigen_2",
        "eigen_3",
        "linearity",
        "planarity",
        "sphericity",
        "omnivariance",
        "anisotropy",
        "eigenentropy",
        "sum_eigen",
        "change_curvature",
        # Spatial descriptors
        "center_x",
        "center_y",
        "center_z",
        "bbox_width",
        "bbox_length",
        "bbox_height",
        "bbox_volume",
        "bbox_yaw",
        # Statistical descriptors
        "num_points",
        "point_density",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "intensity_std",
    ]

    # Regression Targets: Residuals between Proposal and Ground Truth
    # dx, dy, dz: Center offsets
    # dw, dl, dh: Log-scale dimension offsets (log(gt/prop))
    # dyaw: Angle offset
    REGRESSION_TARGETS = ["dx", "dy", "dz", "dw", "dl", "dh", "dyaw"]

    # Class Definitions
    CLASSES = [
        "background",
        "car",
        "truck",
        "other_vehicle",
        "bus",
        "bicycle",
        "pedestrian",
        "motorcycle",
        "animal",
        "emergency_vehicle",
    ]

    # Mappings
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
    ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

    # --------------------------------------------------------------------------
    # MODEL HYPERPARAMETERS (LightGBM)
    # --------------------------------------------------------------------------
    # Training Loop
    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50

    # Classifier Parameters
    LGBM_CLS_PARAMS = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "n_jobs": NUM_WORKERS,
    }

    # Regressor Parameters
    # Note: LightGBM trains one tree per target for regression if using sklearn wrapper,
    # or we train separate models. These are base params.
    LGBM_REG_PARAMS = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "n_jobs": NUM_WORKERS,
    }

    # Inference
    CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to include in submission

    @staticmethod
    def set_seed(seed=None):
        """
        Sets the random seed for reproducibility across Python, Numpy, and Torch.
        """
        if seed is None:
            seed = Config.RANDOM_SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
