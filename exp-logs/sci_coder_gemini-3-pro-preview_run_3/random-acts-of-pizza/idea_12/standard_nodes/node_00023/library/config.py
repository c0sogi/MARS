import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_12")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================
SEED = 42
N_FOLDS = 5

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"
TEXT_COL = "request_text_edit_aware"  # Use edit-aware text to prevent leakage
TITLE_COL = "request_title"
SUBREDDIT_COL = "requester_subreddits_at_request"

# Suffix for columns that constitute data leakage (retrieval time stats)
RETRIEVAL_SUFFIX = "_at_retrieval"

# Columns to explicitly exclude from numerical features (IDs, raw text, etc.)
EXCLUDE_COLS = [
    "requester_received_pizza",
    "request_text",
    "request_text_edit_aware",
    "request_title",
    "requester_subreddits_at_request",
    "request_id",
    "requester_username",
    "source_file",
    "giver_username_if_known",
    "requester_user_flair",
    "post_was_edited",
]

# Names for generated text complexity features
FEATURE_WORD_COUNT = "text_word_count"
FEATURE_SENT_COUNT = "text_sentence_count"
FEATURE_SENTIMENT = "text_sentiment"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# --- Level 1: Lexical Bagger (Random Forest) ---
# High capacity for sparse text data
RF_ESTIMATORS = 500
TEXT_TFIDF_MAX_FEATURES = 3000
TEXT_TFIDF_NGRAM_RANGE = (1, 2)

# --- Level 1: Behavioral Bagger (Random Forest) ---
# High capacity for sparse subreddit history
SUBREDDIT_TFIDF_MAX_FEATURES = 1000
SUBREDDIT_TFIDF_NGRAM_RANGE = (1, 1)

# --- Level 1: Semantic Booster (XGBoost) ---
# High capacity for dense embedding interactions
XGB_ESTIMATORS = 1000
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_EARLY_STOPPING_ROUNDS = 50

# Semantic Encoding
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Level 2: Meta-Learner (Logistic Regression) ---
META_LEARNER_C = 1.0
