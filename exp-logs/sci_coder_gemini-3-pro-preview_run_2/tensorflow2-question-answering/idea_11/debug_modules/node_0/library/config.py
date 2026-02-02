import os


class Config:
    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching (Parquet/Numpy)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    VOCAB_CACHE_PATH = os.path.join(CACHE_DIR, "vocab.npy")  # Stores word->idx mapping
    EMBEDDING_MATRIX_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_matrix.npy")
    TRAIN_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "test_features.parquet")

    # Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Reproducibility ---
    SEED = 42

    # --- Data Processing Parameters ---
    # Vocabulary
    MAX_VOCAB_SIZE = 60000
    MIN_FREQ = 2
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    MAX_SEQ_LEN_Q = 20  # Question length
    MAX_SEQ_LEN_C = 300  # Candidate long answer length

    # --- Model Architecture ---
    EMBEDDING_DIM = 100  # Dimension for word embeddings
    HIDDEN_DIM = 128  # Hidden size for MLPs
    DROPOUT = 0.3  # Dropout rate for regularization

    # --- Training Hyperparameters ---
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    # Negative Sampling
    NEG_SAMPLING_RATIO = 1  # Number of negative candidates per positive candidate

    # Loss Weights (Multi-task learning)
    LOSS_WEIGHT_RANKING = 1.0
    LOSS_WEIGHT_ATTENTION = 0.5  # Supervision for the attention weights
    LOSS_WEIGHT_YESNO = 0.5

    # --- Inference Parameters ---
    LONG_ANSWER_THRESHOLD = 0.1  # Score threshold to predict a long answer vs BLANK
    SHORT_SPAN_WINDOW = (
        25  # Sliding window size for short answer extraction from attention weights
    )

    # --- Debugging / Development ---
    # Set to True to run on a small subset of data for testing the pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    @classmethod
    def setup(cls):
        """Creates necessary directories for outputs and caching."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
