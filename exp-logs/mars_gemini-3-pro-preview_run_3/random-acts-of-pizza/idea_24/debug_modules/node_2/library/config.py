import os


class Config:
    """
    Configuration for the Hex-View Hybrid-Topology Stacking Solution.
    Defines paths, hyperparameters, and feature engineering constants.
    """

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output / Cache
    CACHE_DIR = "./working/idea_24/"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Definitions
    # -------------------------------------------------------------------------
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"
    TEXT_COL = "request_text_edit_aware"  # Use edit-aware to prevent leakage

    # Explicit Allow-List for Dense Metadata Features
    # We strictly exclude retrieval-time features and derived text lengths
    # to prevent leakage and noise.
    DENSE_FEATURES = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # -------------------------------------------------------------------------
    # Feature Engineering Parameters
    # -------------------------------------------------------------------------
    # TF-IDF (Sparse Features)
    TFIDF_PARAMS = {
        "max_features": 3000,
        "min_df": 5,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "stop_words": "english",
        "lowercase": True,
    }

    # Dense Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Manifold Learning
    PCA_COMPONENTS = 50

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Level 1 Base Learners)
    # -------------------------------------------------------------------------

    # 1. Sparse Lexical Branch: Random Forest on Text TF-IDF + Metadata
    MODEL_LEXICAL_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Branch: Random Forest on Subreddit History + Metadata
    MODEL_COMMUNITY_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch: XGBoost on Embeddings + Metadata
    # Note: early_stopping_rounds is handled in the training loop logic
    MODEL_SEMANTIC_XGB = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Approx ratio of neg/pos (75/25)
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # 4. Dense Semantic Branch: Random Forest on Embeddings + Metadata
    MODEL_SEMANTIC_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 5. Manifold Branch: kNN on PCA-Reduced Embeddings + Metadata
    MODEL_MANIFOLD_KNN = {
        "n_neighbors": 50,
        "weights": "distance",
        "metric": "cosine",
        "n_jobs": -1,
    }

    # 6. Contextual Branch: Logistic Regression on Metadata Only
    MODEL_METADATA_LR = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Level 2 Meta-Learner)
    # -------------------------------------------------------------------------
    MODEL_META_LR = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "random_state": SEED}
