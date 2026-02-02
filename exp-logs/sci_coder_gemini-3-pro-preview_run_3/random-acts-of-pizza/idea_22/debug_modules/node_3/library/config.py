import os


class Config:
    """
    Central configuration for the Hex-View Hybrid-Topology Stacking Ensemble.
    Contains paths, feature engineering settings, and model hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    N_JOBS = 12  # Utilizing available vCPUs
    DEBUG = False  # Toggle for debugging with smaller subsets

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    # Input Metadata (Pre-split stratified data)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (For caching features: .npy, .npz, .parquet)
    WORKING_DIR = "./working/idea_22"

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Definitions
    # -------------------------------------------------------------------------
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns
    # Using edit-aware text to avoid leakage from "EDIT: Thanks for pizza"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Behavioral Columns
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Leakage Prevention: Columns to strictly exclude
    # These features are collected at retrieval time and contain future information
    EXCLUDE_PATTERNS = [
        "_at_retrieval",
        "requester_user_flair",
        "post_was_edited",
        "giver_username_if_known",
    ]

    # -------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # -------------------------------------------------------------------------

    # 1. Sparse Lexical View (TF-IDF on Request Text)
    TFIDF_TEXT_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "stop_words": "english",
        "strip_accents": "unicode",
    }

    # 2. Sparse Behavioral View (TF-IDF on Subreddit History)
    # Treating subreddit list as a document of concepts
    TFIDF_HISTORY_PARAMS = {
        "max_features": 1000,
        "ngram_range": (1, 1),
        "min_df": 2,
        "binary": True,  # Presence is more important than frequency for community membership
        "stop_words": None,
    }

    # 3. Dense Semantic View (SBERT Embeddings)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # 4. Manifold View (PCA on Embeddings)
    # Reducing dimensions for kNN performance and density estimation
    PCA_N_COMPONENTS = 50

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5  # Stratified K-Fold

    # Random Forest (Used for: Lexical Bagger, Community Bagger, Semantic Bagger)
    # Regularized with min_samples_leaf to prevent overfitting on sparse data
    RF_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # XGBoost (Used for: Semantic Booster)
    # Gradient boosting for non-linear semantic decision boundaries
    XGB_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": 0,
        # scale_pos_weight will be calculated dynamically based on training fold balance
    }
    XGB_EARLY_STOPPING_ROUNDS = 50

    # k-Nearest Neighbors (Used for: Manifold Neighbor)
    # Exploits local density in the PCA-reduced semantic space
    KNN_PARAMS = {
        "n_neighbors": 30,
        "weights": "distance",
        "metric": "euclidean",
        "n_jobs": N_JOBS,
    }

    # Logistic Regression (Used for: Metadata Anchor, Meta-Learner)
    # High-bias linear baseline and stacking meta-learner
    LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
        "max_iter": 1000,
    }
