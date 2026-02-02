import os


class Config:
    """
    Configuration for Hex-View Stacking Ensemble with NMF-Based Latent Context Injection.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this solution idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_43")
    MODELS_DIR = os.path.join(CACHE_DIR, "models")
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 12  # Based on 12 vCPUs available

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # Text Processing
    TEXT_COLS = ["request_title", "request_text_edit_aware"]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # NMF Latent Context
    NMF_COMPONENTS = 15

    # Sparse Behavioral History
    MAX_SUBREDDITS_VOCAB = 1000  # Top 1k subreddits for sparse bag-of-concepts

    # ==========================================
    # Model Hyperparameters (Level 1 Base Learners)
    # ==========================================

    # 1. Sparse Lexical Bagger (Random Forest)
    # Trained on TF-IDF of Concatenated Text + Metadata
    LEXICAL_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Bagger (Random Forest)
    # Trained on TF-IDF of Subreddit History + Metadata
    BEHAVIORAL_RF_PARAMS = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 3. Semantic Booster (XGBoost)
    # Trained on Dense Embeddings + Metadata
    # Note: scale_pos_weight should be calculated dynamically per fold
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": NUM_WORKERS,
        "random_state": SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    # 4. Semantic Gradient (LightGBM)
    # Trained on Dense Embeddings + Metadata (Algorithmic Diversity)
    LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": NUM_WORKERS,
        "random_state": SEED,
        "objective": "binary",
        "metric": "auc",
    }

    # 5. Semantic Bagger (Random Forest)
    # Trained on Dense Embeddings + Metadata (Structural Diversity)
    # Regularized depth to prevent memorization of dense noise
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 6. Metadata Anchor (Logistic Regression)
    # Trained on Metadata Only (High Bias Regularizer)
    METADATA_LR_PARAMS = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # ==========================================
    # Model Hyperparameters (Level 2 Meta Learner)
    # ==========================================
    META_LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": SEED,
    }

    @classmethod
    def ensure_directories(cls):
        """Creates necessary directories for cache and submission."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
