import os
import torch


class PathConfig:
    """
    Defines file paths for data, caching, and submission.
    """

    # Input Data (Metadata CSVs)
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"

    # Working Directory for Caching Intermediate Files
    # Using idea_5 as specified in the architecture plan
    WORKING_DIR = "./working/idea_5/"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Prefixes for Data Streams
    # Stream A: Lexical/Sparse Data for Random Forest
    STREAM_A_PREFIX = os.path.join(WORKING_DIR, "stream_a_")
    # Stream B: Semantic/Dense/Index Data for MLP
    STREAM_B_PREFIX = os.path.join(WORKING_DIR, "stream_b_")

    # Submission Output
    SUBMISSION_DIR = "./submission/"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")


class FeatureConfig:
    """
    Defines feature columns and engineering hyperparameters.
    """

    # Key Columns
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Raw Text Columns to be processed
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Community Column
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numerical Columns (Leakage-free: only *_at_request)
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

    # Text Processing - Lexical (RF)
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Text Processing - Semantic (MLP)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # Community Processing
    TOP_K_SUBREDDITS = 1000  # Vocabulary size for subreddit embedding


class ModelConfig:
    """
    Defines hyperparameters for Base Learners and Meta Learner.
    """

    # --- Level-1 Base Learner A: Random Forest ---
    RF_PARAMS = {
        "n_estimators": 500,
        "max_depth": 25,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": 0,
    }

    # --- Level-1 Base Learner B: Triple-Branch MLP ---
    # Input Dimensions
    SEMANTIC_INPUT_DIM = 384  # Output dim of all-MiniLM-L6-v2
    SUBREDDIT_EMBED_DIM = 32  # Dimension for subreddit embeddings

    # Hidden Layer Dimensions
    BRANCH_SEMANTIC_HIDDEN = 128
    BRANCH_META_HIDDEN = 64
    BRANCH_COMMUNITY_HIDDEN = 32
    FUSION_HIDDEN = 64

    # Regularization
    DROPOUT_HIGH = 0.5  # For semantic branch
    DROPOUT_MEDIUM = 0.3  # For fusion
    DROPOUT_LOW = 0.1  # For metadata branch

    # --- Level-2 Meta Learner: Stacking ---
    STACKING_LR_C = 1.0  # Inverse regularization strength for Logistic Regression


class TrainingConfig:
    """
    Defines training runtime parameters.
    """

    SEED = 42

    # Training Loop
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # Cross-Validation
    NUM_FOLDS = 5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
