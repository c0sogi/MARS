import os


class Config:
    # --- Paths ---
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Stratified Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching / Working Files
    # Note: Using .npy for numpy arrays and .parquet for dataframes
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Data Cache
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.parquet")

    # Model Checkpoint
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing Hyperparameters ---
    # Vocabulary
    MAX_VOCAB_SIZE = 30000
    MIN_FREQ = 2
    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    # Sentences are usually short, so we keep this small to optimize the DAN
    MAX_SENT_LEN = 40
    MAX_Q_LEN = 20
    # Limit number of sentences per document to handle memory constraints
    MAX_SENTS_PER_DOC = 300

    # Debugging
    # Set to a small integer (e.g., 1000) to run on a subset, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # --- Model Architecture Hyperparameters ---
    EMBEDDING_DIM = 100  # Dimension for word embeddings
    HIDDEN_DIM = 128  # Hidden dimension for the Deep Averaging Network (DAN) MLP
    DROPOUT_PROB = 0.3  # Dropout probability for regularization
    FREEZE_EMBEDDINGS = True  # Whether to freeze the embedding layer

    # --- Training Hyperparameters ---
    RANDOM_SEED = 42
    BATCH_SIZE = 32  # Number of documents per batch
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2  # Stop if validation loss doesn't improve for N epochs

    # Negative Sampling
    # Ratio of negative sentences (irrelevant) to positive sentences (answers) in training
    NEGATIVE_SAMPLE_RATIO = 4

    # --- Inference Hyperparameters ---
    # Score threshold for predicting a Long Answer.
    # If max sentence score < threshold, predict BLANK.
    CONFIDENCE_THRESHOLD = 0.6

    @staticmethod
    def ensure_dirs():
        """Creates necessary working directories if they don't exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon import
Config.ensure_dirs()
