import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================

# Base directory for metadata (pre-split CSVs)
METADATA_DIR = "./metadata"

# Input Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Caching (Idea 1)
# Used to store processed vocabulary or count tables
CACHE_DIR = "./working/idea_1"
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# MODEL HYPERPARAMETERS (Statistical Sentiment-Relevance Model)
# =============================================================================

# Smoothing parameter for probability estimation (Laplace smoothing)
# P(word|sentiment) = (count(word) + alpha) / (total_counts + alpha * vocab_size)
SMOOTHING_ALPHA = 1.0

# Minimum frequency for a token to be considered in the probability map
# Tokens below this frequency in the training set will be ignored/treated as unknown
MIN_FREQ = 2

# Score Shift (Tau) for Inference
# During inference, we calculate: score = P(selected|word, sentiment) - SCORE_SHIFT
# A positive score implies the word is likely part of the selected span.
# A higher shift makes the model more conservative (selects fewer words).
SCORE_SHIFT = 0.4

# =============================================================================
# DEEP LEARNING HYPERPARAMETERS (RoBERTa)
# =============================================================================

MODEL_NAME = "roberta-base"
MAX_LEN = 96
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "roberta_model_opt_v2.bin")

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================

# If True, processes only a small subset of the data for rapid testing
DEBUG = False
DEBUG_SIZE = 500  # Number of rows to use if DEBUG is True
