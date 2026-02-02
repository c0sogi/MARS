import os


class Config:
    """
    Central configuration for the Multi-View Stacked Generalization Strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata Split Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for intermediate features)
    # Text Embeddings (NPY)
    CACHE_TEXT_TRAIN = os.path.join(WORKING_DIR, "X_text_train.npy")
    CACHE_TEXT_VAL = os.path.join(WORKING_DIR, "X_text_val.npy")
    CACHE_TEXT_TEST = os.path.join(WORKING_DIR, "X_text_test.npy")

    # Metadata Features (Parquet)
    CACHE_META_TRAIN = os.path.join(WORKING_DIR, "df_meta_train.parquet")
    CACHE_META_VAL = os.path.join(WORKING_DIR, "df_meta_val.parquet")
    CACHE_META_TEST = os.path.join(WORKING_DIR, "df_meta_test.parquet")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Feature Configuration
    # ==========================================
    # Transformer model for text embeddings
    TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Numerical columns to be used by the Metadata Expert
    # Selected based on availability at request time
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
        "unix_timestamp_of_request",
    ]

    # Text columns to be concatenated and encoded
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    RANDOM_SEED = 42
    N_FOLDS = 5

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # --- View A: Text Expert ---
    # High regularization (Small C) to handle high-dimensional embeddings (384d)
    TEXT_EXPERT_GRID = {
        "C": [1e-4, 5e-4, 1e-3, 5e-3, 0.01],
        "penalty": ["l2"],
        "solver": ["liblinear"],  # Efficient for high-dimensional data
        "class_weight": ["balanced", None],
    }

    # --- View B: Metadata Expert ---
    # Low regularization (Large C) to exploit strong, low-dimensional signals
    META_EXPERT_GRID = {
        "C": [0.1, 1.0, 10.0, 50.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],  # Standard for low-dimensional data
        "class_weight": ["balanced", None],
    }

    # --- Bagging Configuration ---
    # Applied to both experts to reduce variance
    BAGGING_PARAMS = {
        "n_estimators": 10,
        "max_samples": 0.8,
        "max_features": 1.0,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    # --- Stacking Meta-Learner ---
    # Combines probabilities from View A and View B
    STACKING_META_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }
