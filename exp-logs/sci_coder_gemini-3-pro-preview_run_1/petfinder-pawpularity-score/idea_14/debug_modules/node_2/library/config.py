import os
import numpy as np


class Config:
    """
    Global configuration for the Pawpularity Contest solution.
    Implements settings for the Target-Transformed Stratified Tri-Paradigm Stacking Ensemble.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True for quick debugging runs
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode
    NUM_WORKERS = 4

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for outputs and cache
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_14")

    # Ensure output directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    # Submission path
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Backbones (HuggingFace IDs)
    # =========================================================================
    # Semantic Expert
    MODEL_SIGLIP = "google/siglip-so400m-patch14-384"
    # Geometric Expert
    MODEL_DINOV2 = "facebook/dinov2-large"
    # Textural Expert (Upgraded to V2 for FCMAE pre-training)
    MODEL_CONVNEXTV2 = "facebook/convnextv2-large-1k-224"

    # Native Resolutions
    IMG_SIZE_SIGLIP = 384
    IMG_SIZE_DINOV2 = 518
    IMG_SIZE_CONVNEXT = 224

    # Batch sizes for feature extraction
    BATCH_SIZE_SIGLIP = 32
    BATCH_SIZE_DINOV2 = 16  # Larger model, smaller batch
    BATCH_SIZE_CONVNEXT = 32

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # PCA Components for Tree-based models
    PCA_COMPONENTS = 64

    # Target Transformation Constants (Logit Transform)
    # y' = (y / SCALE) clipped to [MIN, MAX]
    # z = log(y' / (1 - y'))
    TARGET_SCALE = 100.0
    TARGET_MIN = 0.001
    TARGET_MAX = 0.999

    # =========================================================================
    # Cross-Validation
    # =========================================================================
    N_FOLDS = 5
    # Stratification is handled by pre-generated metadata, but we use these settings
    # for any internal CV logic if needed.

    # =========================================================================
    # Level-0 Expert Hyperparameters
    # =========================================================================

    # 1. Ridge Regression (Manifold-Linear, Logit Target)
    # Alphas for RidgeCV
    RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0]

    # 2. Support Vector Regression (Manifold-NonLinear, Logit Target)
    SVR_GRID = {
        "kernel": ["rbf"],
        "C": [1.0, 10.0, 50.0, 100.0],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": ["scale"],
    }

    # 3. ExtraTrees Regressor (Partitioning-Bagging, Raw Target)
    ET_PARAMS = {
        "n_estimators": 500,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 4. LightGBM Regressor (Partitioning-Boosting, Raw Target)
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 2000,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    }

    # =========================================================================
    # Level-1 Meta-Learner Hyperparameters
    # =========================================================================
    # Bayesian Ridge Regressor
    META_MODEL_PARAMS = {
        "max_iter": 300,
        "tol": 1e-3,
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
        "fit_intercept": True,
        "verbose": False,
    }

    # =========================================================================
    # Cache Filenames
    # =========================================================================
    # Features
    CACHE_FEATURES_SIGLIP = os.path.join(IDEA_DIR, "features_siglip.npy")
    CACHE_FEATURES_DINOV2 = os.path.join(IDEA_DIR, "features_dinov2.npy")
    CACHE_FEATURES_CONVNEXT = os.path.join(IDEA_DIR, "features_convnext.npy")

    # IDs (to ensure alignment)
    CACHE_IDS_TRAIN = os.path.join(IDEA_DIR, "ids_train.npy")
    CACHE_IDS_VAL = os.path.join(IDEA_DIR, "ids_val.npy")
    CACHE_IDS_TEST = os.path.join(IDEA_DIR, "ids_test.npy")
