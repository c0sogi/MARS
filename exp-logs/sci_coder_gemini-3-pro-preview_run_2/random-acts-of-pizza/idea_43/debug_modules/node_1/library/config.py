import os


class Config:
    """
    Global configuration for the Hook-Augmented Multi-Field Asymmetric
    Dual-Backbone Ensemble (HAMF-ADBE) pipeline.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_43"
    SUBMISSION_DIR = "./submission"

    # Raw Data Sources
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata Splits
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture: HAMF-ADBE
    # ==========================================
    # Embedding Backbones
    # Anchor: High-resolution, sample-efficient (for Title & Body separately)
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # Auxiliary: Deeper context (for Global & Deep Hook)
    AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Dimensionality Reduction (PCA)
    PCA_GLOBAL_COMPONENTS = 50  # For Global Context (Title + Body)
    PCA_HOOK_COMPONENTS = 20  # For Deep Hook (Title only)

    # Ensemble Configuration
    N_ESTIMATORS = 20  # Number of estimators for BaggingClassifier

    # ==========================================
    # Hyperparameters
    # ==========================================
    # Grid Search Space for Logistic Regression Base Estimator
    PARAM_GRID = {
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "class_weight": ["balanced", None],
    }

    # ==========================================
    # Feature Configuration
    # ==========================================
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Numerical metadata for the 'Robust Metadata' view
    # Includes unix_timestamp_of_request as a critical temporal signal
    NUMERIC_COLS = [
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

    # ==========================================
    # Caching Paths (Numpy Arrays)
    # ==========================================
    # Train Embeddings
    CACHE_TRAIN_ANCHOR_TITLE = os.path.join(WORKING_DIR, "train_anchor_title.npy")
    CACHE_TRAIN_ANCHOR_BODY = os.path.join(WORKING_DIR, "train_anchor_body.npy")
    CACHE_TRAIN_AUX_GLOBAL = os.path.join(WORKING_DIR, "train_aux_global.npy")
    CACHE_TRAIN_AUX_HOOK = os.path.join(WORKING_DIR, "train_aux_hook.npy")

    # Test Embeddings
    CACHE_TEST_ANCHOR_TITLE = os.path.join(WORKING_DIR, "test_anchor_title.npy")
    CACHE_TEST_ANCHOR_BODY = os.path.join(WORKING_DIR, "test_anchor_body.npy")
    CACHE_TEST_AUX_GLOBAL = os.path.join(WORKING_DIR, "test_aux_global.npy")
    CACHE_TEST_AUX_HOOK = os.path.join(WORKING_DIR, "test_aux_hook.npy")

    # Model Checkpoints Directory
    MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "models")
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
