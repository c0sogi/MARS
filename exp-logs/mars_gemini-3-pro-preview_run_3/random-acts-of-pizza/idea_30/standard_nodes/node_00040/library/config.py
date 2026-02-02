import os


class Config:
    """
    Configuration for the Regularized Pent-View Stacking solution.
    Defines paths, global constants, feature engineering settings, and model hyperparameters.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_30"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission File Path
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    N_FOLDS = 5
    N_JOBS = 12  # Utilizing available vCPUs

    # Debugging: Set to an integer (e.g., 100) to limit dataset size during development
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================

    # Text Processing (TF-IDF)
    # High-impact keyword extraction with sublinear scaling
    TFIDF_MAX_FEATURES = 3000
    TFIDF_MIN_DF = 5
    TFIDF_SUBLINEAR = True

    # Semantic Embeddings
    # Compact model to avoid dimensionality curse
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Metadata / Tabular Feature Selection (Positive Selection)
    # Explicitly allow-listing user stats and temporal features.
    # Note: Derived text length features are excluded to reduce noise.
    NUMERICAL_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # ==========================================
    # Model Hyperparameters (Level 1 Base Learners)
    # ==========================================

    # 1. Sparse Lexical Branch (Random Forest)
    # Modality: Text (TF-IDF)
    # Strategy: Capture specific keywords. Regularized with leaf constraints.
    LEXICAL_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,  # Regularization to prevent overfitting on rare words
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Branch (Random Forest)
    # Modality: History (Subreddits TF-IDF)
    # Strategy: Bag-of-Concepts for user history.
    BEHAVIORAL_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch (XGBoost)
    # Modality: Text (Embeddings)
    # Strategy: High capacity gradient boosting with imbalance correction.
    SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Handling ~1:3 class imbalance
        "tree_method": "hist",  # Optimized for dense features
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }
    # Fit parameters for XGBoost (Early Stopping)
    SEMANTIC_XGB_FIT_PARAMS = {"verbose": False}

    # 4. Dense Semantic Branch (Random Forest - Constrained)
    # Modality: Text (Embeddings)
    # Strategy: Strictly constrained depth to prevent memorizing embedding noise.
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,  # Strict depth constraint
        "min_samples_leaf": 4,  # Higher leaf threshold for dense data
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 5. Contextual Branch (Logistic Regression)
    # Modality: Metadata
    # Strategy: High-bias linear anchor.
    METADATA_LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
    }

    # ==========================================
    # Level 2 Meta-Learner
    # ==========================================

    # Strategy: Logistic Regression to calibrate ensemble weights.
    META_LEARNER_PARAMS = {
        "C": 0.1,  # Stronger regularization for the meta-learner
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": None,  # Let probabilities drive the decision
        "random_state": RANDOM_SEED,
    }
