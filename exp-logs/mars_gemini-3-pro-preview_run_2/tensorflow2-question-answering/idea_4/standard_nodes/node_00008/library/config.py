import os


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/Numpy)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # --- Data Parameters ---
    # Max sequence lengths
    MAX_Q_LEN = 30  # Max length for questions
    MAX_C_LEN = 300  # Max length for long answer candidates

    # Vocabulary
    VOCAB_SIZE = 20000  # Max vocabulary size
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # --- Model Hyperparameters ---
    EMBEDDING_DIM = 100  # Dimension of word embeddings
    HIDDEN_SIZE = 128  # Hidden size for Bi-LSTM
    DROPOUT_RATE = 0.2  # Dropout rate

    # --- Training Parameters ---
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2

    # Loss Weights
    # Loss = w_rank * rank_loss + w_span * span_loss + w_class * class_loss
    WEIGHT_RANKING_LOSS = 1.0
    WEIGHT_SPAN_LOSS = 0.5
    WEIGHT_CLASS_LOSS = 0.5

    # Negative Sampling
    # Ratio of negative candidates to positive candidates in a batch
    NEGATIVE_SAMPLE_RATIO = 1

    # --- Inference Parameters ---
    # Threshold for selecting a long answer. If score < threshold, return BLANK.
    LONG_ANSWER_CONFIDENCE_THRESHOLD = 0.5

    # --- Debugging ---
    # Set to a small integer (e.g., 1000) to limit dataset size for quick testing
    # Set to None to use the full dataset
    DEBUG_SAMPLE_SIZE = None

    # Random Seed
    SEED = 42

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
