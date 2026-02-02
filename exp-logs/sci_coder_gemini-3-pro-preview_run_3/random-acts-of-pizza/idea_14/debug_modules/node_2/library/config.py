import os


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"
    TEXT_COL = "request_text"
    TITLE_COL = "request_title"
    # Use the edit-aware text for test/prediction to avoid leakage,
    # though training might use raw text if edit-aware isn't available in train.
    # The pipeline logic should handle the selection, but we define the key here.
    TEXT_EDIT_AWARE_COL = "request_text_edit_aware"

    # ==========================================
    # Paths
    # ==========================================
    # Metadata paths (Input)
    TRAIN_PATH = "./metadata/train.parquet"
    VAL_PATH = "./metadata/val.parquet"
    TEST_PATH = "./metadata/test.parquet"
    SAMPLE_SUBMISSION_PATH = "./input/sampleSubmission.csv"

    # Working directory for caching intermediate files
    WORKING_DIR = "./working/idea_14/"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output path for final submission
    SUBMISSION_DIR = "./submission/"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Lexical View (Sparse Text)
    LEXICAL_VECTORIZER_PARAMS = {
        "ngram_range": (1, 2),
        "max_features": 3000,
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
        "lowercase": True,
    }

    # Behavioral View (Sparse History)
    BEHAVIORAL_VECTORIZER_PARAMS = {
        "ngram_range": (1, 1),
        "max_features": 1000,
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
        "lowercase": True,
        "token_pattern": r"(?u)\b\w+\b",  # Simple token pattern for subreddit names
    }

    # Semantic View (Dense Embeddings)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Generates 384-dim embeddings
    EMBEDDING_BATCH_SIZE = 32

    # ==========================================
    # Model Hyperparameters (Level 1)
    # ==========================================
    # 1. Lexical & 2. Behavioral Baggers (Random Forest)
    RF_PARAMS = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 3. Semantic Booster (XGBoost)
    # Note: scale_pos_weight is usually calculated dynamically based on train set balance
    XGB_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    # 4. Contextual Baseline (Logistic Regression)
    LOGREG_PARAMS = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "liblinear",
        "random_state": RANDOM_SEED,
    }

    # ==========================================
    # Meta-Learner Hyperparameters (Level 2)
    # ==========================================
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }

    # ==========================================
    # Training Configuration
    # ==========================================
    N_FOLDS = 5

    # Columns to exclude from features (Leakage or IDs)
    DROP_COLS = [
        "request_id",
        "requester_username",
        "source_file",
        "requester_received_pizza",
        "request_text",
        "request_title",
        "request_text_edit_aware",
        "requester_subreddits_at_request",
        "post_was_edited",
    ]

    # Suffixes of columns to drop to prevent leakage (retrieval time features)
    LEAKAGE_SUFFIXES = ["_at_retrieval"]
