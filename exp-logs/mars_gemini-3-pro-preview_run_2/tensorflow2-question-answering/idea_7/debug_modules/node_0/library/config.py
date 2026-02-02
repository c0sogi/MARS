import os


class Config:
    # =========================================
    # File Paths & Directories
    # =========================================
    # Input Data
    INPUT_DIR = "./input"
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Caching)
    WORKING_DIR = "./working/idea_7"
    CACHE_DIR = WORKING_DIR  # Alias for clarity

    # Output Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================
    # General Settings
    # =========================================
    SEED = 42
    DEBUG = False  # Set to True to use a smaller subset of data for debugging
    DEBUG_SAMPLE_SIZE = 5000

    # =========================================
    # Data Processing Hyperparameters
    # =========================================
    # Tokenization
    VOCAB_SIZE = 100000  # Max vocabulary size
    MIN_FREQ = 2  # Minimum frequency for a token to be included
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    MAX_Q_LEN = 20  # Maximum question length (tokens)
    MAX_DOC_LEN = 300  # Maximum long answer candidate length (tokens)

    # Caching Filenames
    VOCAB_CACHE_FILE = os.path.join(WORKING_DIR, "vocab.npy")
    EMBEDDING_MATRIX_CACHE_FILE = os.path.join(WORKING_DIR, "embedding_matrix.npy")
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================
    # Model Architecture Hyperparameters
    # =========================================
    EMBEDDING_DIM = 100  # Dimension of word embeddings (e.g., GloVe)
    HIDDEN_DIM = 128  # Hidden dimension for MLPs
    DROPOUT_RATE = 0.3  # Dropout probability

    # Pre-trained Embeddings
    # Assuming we might download or use a provided GloVe file,
    # but for this baseline we might initialize random or use what's available.
    # If using provided packages, we stick to standard dims.

    # =========================================
    # Training Hyperparameters
    # =========================================
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 3  # Early stopping patience

    # Loss Weights (Multi-task learning)
    LOSS_WEIGHT_LONG = 1.0
    LOSS_WEIGHT_SHORT = 1.5  # Give slightly more weight to span prediction
    LOSS_WEIGHT_YESNO = 0.5

    # Negative Sampling
    NEGATIVE_RATIO = (
        1.0  # Ratio of negative candidates to positive candidates per batch
    )

    # =========================================
    # Inference Hyperparameters
    # =========================================
    LONG_CONFIDENCE_THRESHOLD = 0.4  # Threshold for predicting a long answer vs BLANK
    SHORT_CONFIDENCE_THRESHOLD = 0.4  # Threshold for short answer span

    # Yes/No Mapping
    YES_NO_MAP = {0: "NONE", 1: "YES", 2: "NO"}
    NUM_YES_NO_CLASSES = 3
