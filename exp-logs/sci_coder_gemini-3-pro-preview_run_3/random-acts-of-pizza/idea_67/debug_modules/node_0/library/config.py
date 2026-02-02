import os


class Config:
    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_67"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths (Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    RANDOM_SEED = 42
    N_FOLDS = 5
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # =========================================================================
    # FEATURE SELECTION & ENGINEERING
    # =========================================================================
    # Text Columns to be concatenated
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Subreddit Column
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Allow-listed Metadata Columns (Strict Leakage Prevention)
    # Includes User Stats, RAOP History, and Raw Timestamp
    METADATA_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request_utc",
    ]

    # NLP Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SUBREDDIT_VOCAB_SIZE = 1000

    # TF-IDF Settings (Granular Tokenization)
    TFIDF_PARAMS = {
        "sublinear_tf": True,
        "min_df": 2,
        "token_pattern": r"\w{1,}",
        "ngram_range": (1, 2),
        "max_features": 20000,
        "stop_words": "english",
    }

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text) ---

    # 1. Granular Lexical Bagger (Random Forest)
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 2. Lexical Randomizer (ExtraTrees)
    LEXICAL_RANDOMIZER_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "criterion": "gini",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 3. Lexical Anchor (Logistic Regression)
    LEXICAL_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "saga",
        "penalty": "l2",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    # --- Branch 2: Sparse Behavioral (History) ---

    # 4. Community Bagger (Random Forest)
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 5. Community Anchor (Logistic Regression)
    COMMUNITY_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "saga",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    # --- Branch 3: Dense Semantic (Text Embeddings) ---

    # 6. Semantic Booster (XGBoost)
    # Conservative boosting with explicit scale_pos_weight (~3.0 for 25% positive class)
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "scale_pos_weight": 3.0,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    # 7. Semantic Gradient (LightGBM)
    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
        "verbose": -1,
        "early_stopping_rounds": 50,
    }

    # 8. Semantic Bagger (Random Forest)
    # Modality-Specific Regularization
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # --- Branch 4: Contextual (Metadata) ---

    # 9. Metadata Anchor (Logistic Regression)
    METADATA_ANCHOR_PARAMS = {
        "C": 0.1,  # Stronger regularization for low-dimensional feature space
        "solver": "liblinear",
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    # 10. Temporal Booster (LightGBM)
    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
        "verbose": -1,
        "early_stopping_rounds": 50,
    }

    # --- Level 2: Meta-Learner ---

    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": RANDOM_SEED,
    }
