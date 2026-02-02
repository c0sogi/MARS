import os

# Ensure working directories exist immediately upon import
os.makedirs("./working/idea_56", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


class Config:
    """
    Configuration for the Deca-View Full-Spectrum Stacking Ensemble.
    """

    # =========================================================================
    # Global Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_56"
    CACHE_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_SEED = 42
    N_FOLDS = 5

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of rows to use if DEBUG is True

    # =========================================================================
    # Data Definitions & Feature Engineering
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"
    TEXT_COL = "request_text_edit_aware"  # Exclusive use to prevent edit leakage
    TITLE_COL = "request_title"

    # Explicit Allow-List for Metadata Features (Hygienic Feature Selection)
    # Excludes derived text lengths and retrieval-time leakage
    METADATA_COLS = [
        # Raw Timestamp (Temporal Signal)
        "unix_timestamp_of_request_utc",
        # User Statistics (Karma/Activity)
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_subreddits_at_request",
        # Restored RAOP Community History (Valid Priors)
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_days_since_first_post_on_raop_at_request",
    ]

    # Sparse Lexical Settings (TF-IDF)
    TFIDF_PARAMS = {
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "strip_accents": "unicode",
        "analyzer": "word",
        "stop_words": "english",
    }

    # Sparse Behavioral Settings (Community Bag-of-Concepts)
    COMMUNITY_COL = "requester_subreddits_at_request"
    COMMUNITY_VOCAB_SIZE = 1000

    # Dense Semantic Settings (Embeddings)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # =========================================================================
    # Model Hyperparameters (Level 1: Base Learners)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text Modality) ---

    # 1. Lexical Bagger (Random Forest)
    # Concatenation maximizes signal; Leaf regularization prevents overfitting.
    HP_LEXICAL_BAGGER = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # 2. Lexical Randomizer (ExtraTrees)
    # Randomized splits for lower variance and algorithmic diversity.
    HP_LEXICAL_RANDOMIZER = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "max_depth": None,
        "bootstrap": False,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # 3. Lexical Anchor (Logistic Regression)
    # High-bias linear baseline for the strongest modality.
    HP_LEXICAL_ANCHOR = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # --- Branch 2: Sparse Behavioral (History Modality) ---

    # 4. Community Bagger (Random Forest)
    # Treats history as bag-of-concepts; stronger leaf regularization.
    HP_COMMUNITY_BAGGER = {
        "n_estimators": 200,
        "min_samples_leaf": 4,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # 5. Community Anchor (Logistic Regression)
    # Robust linear baseline for sparse history.
    HP_COMMUNITY_ANCHOR = {
        "C": 0.5,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # --- Branch 3: Dense Semantic (Text Modality) ---

    # 6. Semantic Booster (XGBoost)
    # Non-linear signals in continuous space.
    HP_SEMANTIC_BOOSTER = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Approx imbalance ratio
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    # 7. Semantic Gradient (LightGBM)
    # Leaf-wise growth for diversity.
    HP_SEMANTIC_GRADIENT = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "early_stopping_rounds": 50,
    }

    # 8. Semantic Bagger (Random Forest)
    # Modality-specific regularization (depth limit) to prevent dense noise memorization.
    HP_SEMANTIC_BAGGER = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # --- Branch 4: Contextual (Metadata Modality) ---

    # 9. Metadata Anchor (Logistic Regression)
    # High-bias regularizer.
    HP_METADATA_ANCHOR = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # 10. Temporal Booster (LightGBM)
    # Captures non-linear temporal drift via raw timestamps.
    HP_TEMPORAL_BOOSTER = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,  # Small leaves for small feature set
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "early_stopping_rounds": 50,
    }

    # =========================================================================
    # Model Hyperparameters (Level 2: Meta-Learner)
    # =========================================================================
    HP_META_LEARNER = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }

    # =========================================================================
    # Hybrid Inference Protocol
    # =========================================================================
    # True: Stable learner -> Retrain single model on Full Train + Val
    # False: Volatile learner -> Use average of 5 CV models (CV-Bagging)
    RETRAIN_FLAGS = {
        "lexical_bagger": True,
        "lexical_randomizer": True,
        "lexical_anchor": True,
        "community_bagger": True,
        "community_anchor": True,
        "semantic_booster": False,  # Volatile (Early Stopping required)
        "semantic_gradient": False,  # Volatile (Early Stopping required)
        "semantic_bagger": True,
        "metadata_anchor": True,
        "temporal_booster": False,  # Volatile (Early Stopping required)
    }
