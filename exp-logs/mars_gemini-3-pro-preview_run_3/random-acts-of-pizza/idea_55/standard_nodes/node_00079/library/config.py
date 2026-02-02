import os

# Ensure working directory exists for caching
os.makedirs("./working/idea_55", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    WORKING_DIR = "./working/idea_55"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_STATE = 42
    N_FOLDS = 5

    # =========================================================================
    # Feature Columns
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Inputs: Title and Edit-Aware Body
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Community Input: List of subreddits
    COMMUNITY_COL = "requester_subreddits_at_request"

    # Metadata Allow-List (Dense Features)
    # Selected to restore valid domain priors and exclude retrieval-time leakage
    METADATA_COLS = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_days_since_first_post_on_raop_at_request",
    ]

    # =========================================================================
    # Preprocessing Configuration
    # =========================================================================
    # TF-IDF for Text (Sparse Lexical)
    TEXT_VECTORIZER_PARAMS = {
        "sublinear_tf": True,
        "min_df": 5,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "max_features": 20000,
    }

    # TF-IDF for Community (Sparse Behavioral / Bag-of-Concepts)
    COMMUNITY_VECTORIZER_PARAMS = {
        "max_features": 1000,  # Top 1000 subreddits
        "binary": True,  # Presence/Absence
        "stop_words": None,
        "use_idf": True,
        "norm": "l2",
    }

    # Semantic Embeddings (Dense Text)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # =========================================================================
    # Model Hyperparameters (Level 1)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text) ---
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    LEXICAL_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "penalty": "l2",
        "random_state": RANDOM_STATE,
    }

    # --- Branch 2: Sparse Behavioral (History) ---
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    COMMUNITY_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "penalty": "l2",
        "random_state": RANDOM_STATE,
    }

    # --- Branch 3: Dense Semantic (Text Embeddings) ---
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Handle ~3:1 imbalance
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "enable_categorical": False,
    }

    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # --- Branch 4: Contextual (Metadata) ---
    METADATA_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }

    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    # =========================================================================
    # Model Hyperparameters (Level 2)
    # =========================================================================
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # =========================================================================
    # Hybrid Inference Protocol
    # =========================================================================
    # STABLE: Retrain single model on Train + Val (Maximize Data)
    STABLE_MODELS = [
        "lexical_bagger",
        "lexical_anchor",
        "community_bagger",
        "community_anchor",
        "semantic_bagger",
        "metadata_anchor",
    ]

    # VOLATILE: Use CV-Bagging (Average of K-Fold models) to respect Early Stopping
    VOLATILE_MODELS = ["semantic_booster", "semantic_gradient", "temporal_booster"]
