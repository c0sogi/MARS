import os
import torch
import numpy as np


class Config:
    """
    Global configuration for the Pawpularity Contest solution.
    Implements settings for:
    - Stratified Tri-Paradigm Stacking Ensemble
    - 3 Backbones (SigLIP, DINOv2, ConvNeXt)
    - 4 Level-0 Experts (Ridge, SVR, ExtraTrees, LightGBM)
    - 1 Level-1 Meta-Learner (Bayesian Ridge)
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Compute / Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Generated Metadata Paths (from the metadata generation step)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching (Embeddings, Models)
    # Using 'idea_11' as the current iteration workspace
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = WORKING_DIR

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Cross-Validation Strategy
    # =========================================================================
    N_FOLDS = 5
    # Number of bins to discretize the continuous target for stratified splitting
    STRATIFY_BINS = 15

    # =========================================================================
    # Feature Extraction (Backbones)
    # =========================================================================
    # Dictionary defining the pre-trained models to be used for feature extraction.
    # Keys act as identifiers for caching.
    BACKBONES = {
        "siglip": {
            "model_id": "google/siglip-so400m-patch14-384",
            "batch_size": 32,
            "target_size": (384, 384),
            "description": "Semantic Expert",
        },
        "dinov2": {
            "model_id": "facebook/dinov2-large",
            "batch_size": 32,
            "target_size": (518, 518),
            "description": "Geometric Expert",
        },
        "convnext": {
            "model_id": "facebook/convnext-large-224-22k-1k",
            "batch_size": 32,
            "target_size": (224, 224),
            "description": "Textural Expert",
        },
    }

    # =========================================================================
    # Level-0 Experts (Hyperparameters)
    # =========================================================================
    # PCA Components for Tree/Boosting models to reduce dimensionality of embeddings
    PCA_COMPONENTS = 64

    # 1. Ridge Regression (Linear Expert)
    # Log-spaced alphas to cover various degrees of regularization
    RIDGE_ALPHAS = np.logspace(-2, 5, 20).tolist()

    # 2. Support Vector Regression (Kernel Expert)
    SVR_PARAMS = {
        "kernel": "rbf",
        "C": [0.1, 1.0, 10.0, 50.0],
        "epsilon": [0.01, 0.1, 0.5],
        "cache_size": 2000,
    }

    # 3. Extra Trees Regressor (Bagging Expert)
    ET_PARAMS = {
        "n_estimators": 500,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 4. LightGBM Regressor (Boosting Expert)
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": SEED,
        "force_col_wise": True,
    }

    # =========================================================================
    # Level-1 Meta-Learner
    # =========================================================================
    # Bayesian Ridge Regressor parameters
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

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        Should be called at the start of the execution pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_lgbm_params(cls, override_params=None):
        """
        Returns LightGBM parameters, optionally updating with overrides.
        """
        params = cls.LGBM_PARAMS.copy()
        if override_params:
            params.update(override_params)
        return params
