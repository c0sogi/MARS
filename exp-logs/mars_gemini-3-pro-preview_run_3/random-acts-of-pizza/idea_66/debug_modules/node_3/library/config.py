import os


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    RANDOM_SEED = 42
    N_FOLDS = 5
    N_JOBS = 12  # Utilizing available vCPUs

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    # Input data (using pre-generated metadata parquets)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"  # Not strictly used if we merge for Union Dataset, but good to have
    TEST_DATA_PATH = "./metadata/test.parquet"

    # Cache directory for intermediate artifacts (embeddings, processed matrices)
    CACHE_DIR = "./working/idea_66"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Feature Selection & Engineering
    # =========================================================================
    # Target Column
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Columns
    TEXT_TITLE_COL = "request_title"
    TEXT_BODY_COL = (
        "request_text_edit_aware"  # Use edit-aware version to prevent leakage
    )

    # History Column (List of subreddits)
    HISTORY_COL = "requester_subreddits_at_request"

    # Allow-Listed Metadata Features (Strict Leakage Prevention)
    # We exclude all columns ending in '_at_retrieval'
    # We include User Stats and Raw Timestamp
    METADATA_COLS = [
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

    # =========================================================================
    # Vectorization & Embeddings (Differential Sparsity)
    # =========================================================================

    # 1. Open-Vocabulary Lexical Branch (Text)
    # Granular tokenization to capture agency tokens, no max_features for long tail
    TEXT_TFIDF_PARAMS = {
        "token_pattern": r"\w{1,}",  # Capture single char tokens like 'I', 'a'
        "min_df": 2,  # Prune extremely rare typos
        "sublinear_tf": True,  # Log scaling
        "ngram_range": (1, 2),  # Unigrams and Bigrams
        "max_features": None,  # Open vocabulary
    }

    # 2. Closed-Vocabulary Behavioral Branch (History)
    # Constrained to prevent overfitting to rare subreddits
    HISTORY_TFIDF_PARAMS = {
        "max_features": 1000,  # Closed vocabulary constraint
        "binary": True,  # Presence/Absence is more important than count
        "norm": "l2",
    }

    # 3. Dense Semantic Branch
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim dense vectors

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # --- Level 1: Base Learners ---

    # 1. Lexical Branch (Text)
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,  # Slight regularization for sparse data
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    LEXICAL_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # 2. Behavioral Branch (History)
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 200,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        # Default depth for closed vocab is usually fine, or slight restriction
    }

    COMMUNITY_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # 3. Semantic Branch (Dense)
    # Conservative Boosting for XGB
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,  # Conservative
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,  # Feature subsampling
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",  # Efficient for dense data
        "early_stopping_rounds": 100,
        # scale_pos_weight will be calculated dynamically
    }

    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    }

    # Modality-Specific Regularization for RF on Dense Data
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,  # Prevent memorization of dense noise
        "min_samples_leaf": 4,  # High leaf constraint
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    # 4. Contextual Branch (Metadata)
    METADATA_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,  # Smaller trees for low-dim data
        "n_jobs": N_JOBS,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "early_stopping_rounds": 50,
    }

    # --- Level 2: Meta Learner ---
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
        # No class_weight here usually, as inputs are probabilities
    }
