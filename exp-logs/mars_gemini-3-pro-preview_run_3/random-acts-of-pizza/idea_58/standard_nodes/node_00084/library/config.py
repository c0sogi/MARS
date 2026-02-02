import os


class Config:
    # --------------------------------------------------------------------------
    # Global Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5
    # Debugging / Development flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100 if DEBUG else None

    # --------------------------------------------------------------------------
    # Directories and Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_58"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Feature Engineering Configuration
    # --------------------------------------------------------------------------
    # Granular Tokenization Regex: Captures single characters (e.g., 'I', '$')
    GRANULAR_TOKEN_PATTERN = r"\w{1,}"

    # Pre-trained Embedding Model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Metadata Allow-List: Explicitly allowed numerical/temporal features
    META_FEATURES_ALLOWLIST = [
        "unix_timestamp_of_request_utc",
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

    # Columns to Exclude (Target, IDs, Raw Text, Leakage)
    DROP_COLS = [
        "requester_received_pizza",
        "request_id",
        "request_text",
        "request_text_edit_aware",
        "request_title",
        "requester_subreddits_at_request",
        "source_file",
        "requester_username",
        "giver_username_if_known",
        "post_was_edited",
        "requester_user_flair",
    ]

    # Suffix for retrieval-time leakage columns
    LEAKAGE_SUFFIX = "_at_retrieval"

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # 1. Sparse Lexical Branch: Granular Lexical Bagger (Random Forest)
    LEXICAL_VECTORIZER_PARAMS = {
        "min_df": 5,
        "sublinear_tf": True,
        "token_pattern": GRANULAR_TOKEN_PATTERN,
        "ngram_range": (1, 2),
        "max_features": 20000,
    }
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "random_state": SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 2. Sparse Behavioral Branch: Community Bagger (Random Forest)
    COMMUNITY_VECTORIZER_PARAMS = {
        "max_features": 1000,  # Limit to top 1000 subreddits
        "binary": True,
        "token_pattern": r"(?u)\b\w+\b",
    }
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 200,
        "random_state": SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 3. Dense Semantic Branch: Semantic Booster (XGBoost)
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,  # Strict regularization
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
    }

    # 4. Dense Semantic Branch: Semantic Gradient (LightGBM)
    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
        "metric": "binary_logloss",
        "early_stopping_rounds": 50,
    }

    # 5. Dense Semantic Branch: Semantic Bagger (Random Forest)
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,  # Modality-specific regularization
        "min_samples_leaf": 4,
        "random_state": SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 6. Contextual Branch: Metadata Anchor (Logistic Regression)
    METADATA_ANCHOR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": SEED,
        "class_weight": "balanced",
    }

    # 7. Contextual Branch: Temporal Booster (LightGBM)
    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
        "metric": "binary_logloss",
        "early_stopping_rounds": 50,
    }

    # Level 2: Meta-Learner (Logistic Regression)
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": SEED,
    }
