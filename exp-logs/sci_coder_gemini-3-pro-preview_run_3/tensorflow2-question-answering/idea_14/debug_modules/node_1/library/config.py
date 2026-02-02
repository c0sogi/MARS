import os

# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Raw Data
TRAIN_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
TEST_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Pre-generated)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Caching Paths (Parquet/Numpy for intermediate data)
VOCAB_CACHE_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
EMBEDDING_MATRIX_CACHE_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

# Processed Datasets Cache
RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

# Pre-computed features for Test set (to save time during inference if re-run)
RANKER_TEST_INPUTS_CACHE = os.path.join(WORKING_DIR, "ranker_test_inputs.parquet")

# Model Checkpoints
RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

# Output Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Global Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42

# Data Processing
MAX_VOCAB_SIZE = 30000  # Limit vocabulary size to top frequent words
EMBEDDING_DIM = 100  # Dimension for word embeddings (e.g., GloVe-like)
MAX_Q_LEN = 30  # Maximum length for questions
MAX_DOC_LEN = 256  # Maximum length for candidate paragraphs
MIN_DOC_LEN = 10  # Minimum length to consider a paragraph valid

# Model Architecture: K-Max Interaction Ranker
K_MAX = 5  # Number of top interactions to pool per query token

# Model Architecture: Highway Co-Attention Reader
HIGHWAY_LAYERS = 2  # Depth of highway network
HIDDEN_DIM = 128  # Hidden dimension size for internal layers

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 2  # Stop if validation loss doesn't improve for 2 epochs

# Inference
CONFIDENCE_THRESHOLD = 0.4  # Threshold for non-null predictions
NULL_PREDICTION_STRING = ""  # String representation for no answer

# Debugging / Development
# Set to an integer (e.g., 5000) to limit dataset size for fast debugging.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = 10000
