import os


class Config:
    """
    Global configuration for the Modality-Balanced Bagged Linear Ensemble project.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet for tabular, NPY for embeddings)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
    VAL_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "val_embeddings.npy")
    TEST_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "test_embeddings.npy")

    # ==========================================
    # Model & Feature Configuration
    # ==========================================
    # Text Embedding
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Text Columns to Concatenate
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numeric Metadata Features
    # Explicitly including unix_timestamp_of_request as per strategy
    NUMERIC_FEATURES = [
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
    # Training & Hyperparameters
    # ==========================================
    N_SPLITS = 5
    BAGGING_N_ESTIMATORS = 10

    # Grid Search Space
    GRID_SEARCH_PARAMS = {
        # Regularization strength (Inverse of regularization)
        # Focus on High-Regularization Regime (small C) for noisy text embeddings
        # Capped at 10.0 to prevent overfitting (Cite solution_lesson_node_00022)
        "C": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0],
        # Class Weights to handle imbalance (Cite solution_lesson_node_00028)
        "class_weight": ["balanced", None],
    }
