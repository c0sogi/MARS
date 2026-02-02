import os


class Config:
    """
    Configuration for the Hex-View Stacking Ensemble.
    """

    # ---------------------------------------------------------
    # General Settings
    # ---------------------------------------------------------
    RANDOM_SEED = 42
    N_JOBS = 12  # Number of vCPUs available

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directory for caching processed features
    WORKING_DIR = "./working/idea_39"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Data Definitions
    # ---------------------------------------------------------
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text columns to be concatenated
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Columns to exclude from Metadata Vector (Leakage prevention + Raw Text)
    # Note: Columns ending in '_at_retrieval' are excluded dynamically in the pipeline.
    EXCLUDE_COLS = [
        "requester_received_pizza",
        "request_id",
        "requester_username",
        "source_file",
        "request_text",
        "request_title",
        "request_text_edit_aware",
        "requester_subreddits_at_request",
        "giver_username_if_known",
        "requester_user_flair",
        "post_was_edited",
    ]

    # ---------------------------------------------------------
    # Feature Engineering Parameters
    # ---------------------------------------------------------
    # Text Vectorization (Lexical Branch)
    TEXT_VEC_PARAMS = {
        "min_df": 5,
        "sublinear_tf": True,
        "ngram_range": (1, 2),
        "max_features": 10000,
        "stop_words": "english",
    }

    # Subreddit Vectorization (Behavioral Branch)
    SUBREDDIT_VEC_PARAMS = {
        "min_df": 1,
        "max_features": 1000,  # Strict limit as per design
        "binary": True,  # Bag-of-Concepts approach
    }

    # Dense Embeddings (Semantic Branch)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Interaction Branch (Low-Rank SVD)
    SVD_N_COMPONENTS_TEXT = 32
    SVD_N_COMPONENTS_HISTORY = 32

    # ---------------------------------------------------------
    # Model Hyperparameters (Level 1 Base Learners)
    # ---------------------------------------------------------

    # 1. Lexical Bagger (Random Forest on Sparse Text + Meta)
    MODEL_LEXICAL_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 2. Community Bagger (Random Forest on Sparse History + Meta)
    MODEL_COMMUNITY_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 3. Semantic Booster (XGBoost on Dense Embeddings + Meta)
    MODEL_SEMANTIC_XGB = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": 3.0,  # Handling imbalance
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "early_stopping_rounds": 100,
        "eval_metric": "auc",
        "verbosity": 0,
    }

    # 4. Semantic Bagger (Random Forest on Dense Embeddings + Meta)
    MODEL_SEMANTIC_RF = {
        "n_estimators": 300,
        "max_depth": 12,  # Regularization for dense inputs
        "min_samples_leaf": 4,  # Regularization for dense inputs
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 5. Interaction Booster (XGBoost on Low-Rank SVD + Meta)
    MODEL_INTERACTION_XGB = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "early_stopping_rounds": 100,
        "eval_metric": "auc",
        "verbosity": 0,
    }

    # 6. Metadata Anchor (Logistic Regression on Meta only)
    MODEL_METADATA_LR = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
    }

    # ---------------------------------------------------------
    # Model Hyperparameters (Level 2 Meta-Learner)
    # ---------------------------------------------------------
    MODEL_META_LR = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
    }
