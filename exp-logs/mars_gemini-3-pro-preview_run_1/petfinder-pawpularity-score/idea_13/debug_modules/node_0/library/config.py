import os
import numpy as np


class Config:
    """
    Global configuration for the Pawpularity Contest pipeline.
    Implements the 'Stratified Tri-Paradigm Stacking with Interaction-Aware Meta-Learning' strategy.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 2  # Adjust based on vCPU availability (12 vCPUs available)

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Cache directory for deterministic data processing (e.g., embeddings)
    # As per requirements, using ./working/idea_13/
    CACHE_DIR = "./working/idea_13"

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Schema
    # =========================================================================
    ID_COL = "Id"
    TARGET_COL = "Pawpularity"
    FILE_PATH_COL = "file_path"

    # The 12 binary metadata features used for Interaction-Aware Meta-Learning
    META_FEATURES = [
        "Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # =========================================================================
    # Feature Extraction (Backbones)
    # =========================================================================
    # Dictionary mapping friendly names to Hugging Face Hub IDs
    BACKBONES = {
        # Semantic Expert: High performance on subjective tasks
        "siglip": "google/siglip-so400m-patch14-384",
        # Geometric Expert: Self-supervised layout/structure understanding
        "dinov2": "facebook/dinov2-large",
        # Textural Expert: High-frequency local texture capture
        "convnext": "facebook/convnext-large-224-22k-1k",
    }

    # Inference settings
    BATCH_SIZE = 32

    # Feature-Space Augmentation: Average embeddings of original and horizontally flipped images
    USE_FLIP_AUGMENTATION = True

    # =========================================================================
    # Level-0 Experts Configuration
    # =========================================================================
    # Dimensionality reduction for tree-based models to mitigate curse of dimensionality
    PCA_COMPONENTS = 64

    # 1. Linear Expert: Ridge Regression
    # Search range for alpha (regularization strength)
    RIDGE_ALPHAS = np.logspace(-1, 5, 20).tolist()

    # 2. Kernel Expert: Support Vector Regression (RBF)
    SVR_PARAMS = {
        "kernel": "rbf",
        "C": 10.0,  # Regularization parameter
        "epsilon": 0.1,  # Epsilon in the epsilon-SVR model
        "gamma": "scale",
        "cache_size": 2000,
    }

    # 3. Bagging Expert: ExtraTrees Regressor
    ET_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 4. Boosting Expert: LightGBM Regressor
    LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.005,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "metric": "rmse",
    }
    # Early stopping rounds for LightGBM
    LGBM_ES_ROUNDS = 100

    # =========================================================================
    # Level-1 Meta-Learner Configuration
    # =========================================================================
    # Bayesian Ridge Regression for handling high-dimensional interaction terms
    # (Predictions + Metadata + Predictions*Metadata)
    META_MODEL_PARAMS = {
        "n_iter": 300,
        "tol": 1e-3,
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
        "fit_intercept": True,
        "compute_score": True,
        "verbose": False,
    }

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
